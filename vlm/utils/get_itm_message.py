import cv2
import numpy as np
from vlm.itm.blip2itm import BLIP2ITMClient
from vlm.config import VLM_BLIP2ITM_PORT

itmclient = BLIP2ITMClient(port=VLM_BLIP2ITM_PORT)

def get_itm_message(rgb_image, label):
    txt = f"Is there a {label} in the image?"
    cosine = itmclient.cosine(rgb_image, txt)
    itm_score = itmclient.itm_score(rgb_image, txt)
    return cosine, itm_score

def get_itm_message_cosine(rgb_image, label, room):
    if room != "everywhere":
        txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        txt = f"Seems like there is a {label} ahead?"
    cosine = itmclient.cosine(rgb_image, txt)
    return cosine