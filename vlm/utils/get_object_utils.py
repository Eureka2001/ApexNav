import cv2
import numpy as np

from vlm.coco_classes import COCO_CLASSES
from vlm.config import (
    VLM_BLIP2ITM_PORT,
    VLM_GROUNDING_DINO_PORT,
    VLM_MOBILE_SAM_PORT,
    VLM_YOLOV7_PORT,
)
from vlm.detector.grounding_dino import GroundingDINOClient
from vlm.detector.yolov7 import YOLOv7Client
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.segmentor.sam import MobileSAMClient
from vlm.utils.get_itm_message import get_itm_message

yolov7_detector = YOLOv7Client(port=VLM_YOLOV7_PORT)
blip2_itm = BLIP2ITMClient(port=VLM_BLIP2ITM_PORT)
sam_segmentor = MobileSAMClient(port=VLM_MOBILE_SAM_PORT)
dino_detector = GroundingDINOClient(port=VLM_GROUNDING_DINO_PORT)


def get_segmentation(segmented_img, idx, detections, img, label, score, color):
    object_mask = np.zeros((480, 640), dtype=np.uint8)
    bbox_denorm = detections.boxes[idx] * np.array(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    )
    x1, y1, x2, y2 = [int(v) for v in bbox_denorm]
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]

    if bbox_area / img_area < 0.99:
        object_mask = sam_segmentor.segment_bbox(img, bbox_denorm.tolist())
        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cv2.drawContours(segmented_img, [contour], 0, color, 4)

        cv2.rectangle(
            segmented_img,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
        )
        label_x = x1
        label_y = y1 - text_height
        cv2.rectangle(
            segmented_img,
            (label_x, label_y - 30),
            (label_x + text_width, label_y + text_height),
            color,
            2,
        )
        cv2.putText(
            segmented_img,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
        )

    return segmented_img, object_mask


def get_object(right_label, img, cfg, similar_answer):
    score_list = []
    object_masks_list = []
    segmented_img = img.copy()
    label_list = []
    coco_label = []
    dino_label = []
    right_label_list = list(map(str.strip, right_label.split("|")))
    # print(f"right_label_list: {right_label_list}")
    all_answer = right_label_list + similar_answer
    for label in all_answer:
        if label in COCO_CLASSES:
            coco_label.append(label)
        else:
            dino_label.append(label)

    if any(item in dino_label for item in right_label_list):
        dino_label = all_answer
        coco_label = []
        for label in right_label_list:
            if label in COCO_CLASSES:
                coco_label.append(label)

    if coco_label:
        detections = yolov7_detector.predict(
            img,
            agnostic_nms=cfg.yolo.agnostic_nms,
            conf_thres=cfg.yolo.confidence_threshold_yolo,
            iou_thres=cfg.yolo.iou_threshold_yolo,
        )
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if detections.phrases[idx] in right_label_list:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    label_detected,
                    score,
                    color=(255, 0, 0),
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(0)
            elif detections.phrases[idx] in coco_label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    label_detected,
                    score,
                    color=(0, 255, 0),
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(
                    list(all_answer).index(label_detected) - len(right_label_list) + 1
                )

    if dino_label:
        caption = " ".join(f"{item}.  " for item in dino_label)
        detections = dino_detector.predict(
            img,
            caption=caption,
            box_threshold=cfg.groundingDINO.confidence_threshold_dino,
            text_threshold=cfg.groundingDINO.text_threshold,
        )
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if label_detected in right_label_list:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    label_detected,
                    score,
                    color=(255, 0, 0),
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(0)

            elif label_detected in dino_label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    label_detected,
                    score,
                    color=(0, 255, 0),
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                label_list.append(
                    list(all_answer).index(label_detected) - len(right_label_list) + 1
                )

    return segmented_img, score_list, object_masks_list, label_list


def get_object_with_itm(label, img, cfg):
    score_list = []
    object_masks_list = []
    cosine_list = []
    itm_score_list = []
    segmented_img = img.copy()
    if label in COCO_CLASSES:
        detections = yolov7_detector.predict(
            img,
            agnostic_nms=cfg.yolo.agnostic_nms,
            conf_thres=cfg.yolo.confidence_threshold_yolo,
            iou_thres=cfg.yolo.iou_threshold_yolo,
        )
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if detections.phrases[idx] == label:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    label_detected,
                    score,
                    color=(255, 0, 0),
                )
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                print(f"cosine: {cosine:.3f}, itm_score: {itm_score:.3f}")
                score_list.append(score)
                object_masks_list.append(object_mask)
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)

    else:
        detections = dino_detector.predict(
            img,
            caption=label,
            box_threshold=cfg.groundingDINO.confidence_threshold_dino,
            text_threshold=cfg.groundingDINO.text_threshold,
        )
        for idx in range(len(detections.logits)):
            label_detected = detections.phrases[idx]
            score = detections.logits[idx].item()
            if score > cfg.groundingDINO.confidence_threshold_dino:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    label_detected,
                    score,
                    color=(255, 0, 0),
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                img_detected = crop_and_expand_box(img, detections, idx)
                # cv2.imshow(f"img_detected{idx}", img_detected)
                cosine, itm_score = get_itm_message(img_detected, label)
                print(f"cosine: {cosine}, itm_score: {itm_score}")
                cosine_list.append(cosine)
                itm_score_list.append(itm_score)

    return segmented_img, score_list, object_masks_list, cosine_list, itm_score_list


def crop_and_expand_box(img, detections, idx, expand_pixels=0.4):
    # 获取框的坐标，格式为 [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = detections.boxes[idx]
    x_min = int(x_min * img.shape[1])
    y_min = int(y_min * img.shape[0])
    x_max = int(x_max * img.shape[1])
    y_max = int(y_max * img.shape[0])

    # 向外扩展框，注意不要超出图像边界
    x_min = max(int(x_min * (1 - expand_pixels)), 0)
    y_min = max(int(y_min * (1 - expand_pixels)), 0)
    x_max = min(int(x_max * (1 + expand_pixels)), img.shape[1] - 1)
    y_max = min(int(y_max * (1 + expand_pixels)), img.shape[0] - 1)

    # 裁剪图像，仅保留框内内容
    img_detected = img[y_min : y_max + 1, x_min : x_max + 1]

    return img_detected


def detect_objects(object_classes, img, cfg):
    """
    检测列表中的所有物体并返回结果

    参数:
        object_classes: 待检测物体类别列表
        img: 输入图像
        cfg: 检测器配置参数

    返回:
        segmented_img: 带检测结果可视化的图像
        score_list: 检测分数列表
        object_masks_list: 物体掩码列表
        class_indices: 检测到的物体类别索引列表
    """
    score_list = []
    object_masks_list = []
    class_indices = []
    segmented_img = img.copy()

    # 将类别分为COCO标准类别和自定义类别
    coco_classes = []
    custom_classes = []
    for cls in object_classes:
        if cls in COCO_CLASSES:
            coco_classes.append(cls)
        else:
            custom_classes.append(cls)

    # 检测COCO标准类别
    if coco_classes:
        detections = yolov7_detector.predict(
            img,
            agnostic_nms=cfg.yolo.agnostic_nms,
            conf_thres=cfg.yolo.confidence_threshold_yolo,
            iou_thres=cfg.yolo.iou_threshold_yolo,
        )

        for idx in range(len(detections.logits)):
            class_name = detections.phrases[idx]
            score = detections.logits[idx].item()

            # 只保留在目标类别列表中的检测结果
            if class_name in coco_classes:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    class_name,
                    score,
                    color=(0, 255, 0),
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                class_indices.append(object_classes.index(class_name))

    # 检测自定义类别
    if custom_classes:
        # 构建GroundingDINO所需的文本提示
        caption = " ".join(f"{cls}.  " for cls in custom_classes)
        detections = dino_detector.predict(
            img,
            caption=caption,
            box_threshold=cfg.groundingDINO.confidence_threshold_dino,
            text_threshold=cfg.groundingDINO.text_threshold,
        )

        for idx in range(len(detections.logits)):
            class_name = detections.phrases[idx]
            score = detections.logits[idx].item()

            # 只保留在目标类别列表中的检测结果
            if class_name in custom_classes:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    class_name,
                    score,
                    color=(255, 0, 0),
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                class_indices.append(object_classes.index(class_name))

    return segmented_img, score_list, object_masks_list, class_indices
