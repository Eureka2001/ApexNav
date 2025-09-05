import cv2
import numpy as np
import tf
from tf.transformations import quaternion_from_euler

from value_map.config import COLOR_MAP_FOR_OBJECT, COMMON_OBJECTS


class MultiSemanticMap:
    def __init__(self, map_size=480, resolution=0.05, channel_num=20):
        self.map_size = map_size
        self.resolution = resolution
        self.channel_num = channel_num

        self.label_to_index = {label: idx for idx, label in enumerate(COMMON_OBJECTS)}
        if self.channel_num < len(COMMON_OBJECTS):
            self.channel_num = len(COMMON_OBJECTS)

        self.map_center_x = 0.0
        self.map_center_y = 0.0

        self.map = np.zeros(
            (self.channel_num, self.map_size, self.map_size), dtype=np.float32
        )
        self.update_times = np.zeros(
            (self.channel_num, self.map_size, self.map_size), dtype=np.uint32
        )

    @staticmethod
    def calulate_tmat_camera2world(gps, compass, pitch):
        """
        计算从相机坐标系到世界坐标系的变换矩阵，与 habitat_publisher.py 中的逻辑一致，逻辑比较特殊，不可随意复用。

        Args:
            gps: GPS坐标 [x, y, z]
            compass: 罗盘角度
            pitch: 俯仰角

        Returns:
            tmat_camera2world: 4x4变换矩阵
        """
        # 创建4x4的变换矩阵
        tmat_camera2world = np.eye(4)

        # 设置平移部分
        tmat_camera2world[0, 3] = -gps[2]
        tmat_camera2world[1, 3] = -gps[0]
        tmat_camera2world[2, 3] = gps[1] + 0.88  # 0.88是相机高度偏移

        # 计算旋转部分
        quat = quaternion_from_euler(pitch + np.pi / 2.0, np.pi, compass + np.pi / 2.0)
        rotation_matrix = tf.transformations.quaternion_matrix(quat)

        tmat_camera2world[:3, :3] = rotation_matrix[:3, :3]
        return tmat_camera2world

    def _world_to_map_coords(self, points_world):
        """将世界坐标转换为地图坐标"""
        map_coords = np.zeros_like(points_world[:, :2])
        map_coords[:, 0] = (
            points_world[:, 0] - self.map_center_x
        ) / self.resolution + self.map_size / 2
        map_coords[:, 1] = (
            points_world[:, 1] - self.map_center_y
        ) / self.resolution + self.map_size / 2
        return map_coords.astype(int)

    def _get_channel_index(self, label):
        """
        根据标签获取通道索引
        """
        if isinstance(label, str):
            if label in self.label_to_index:
                return self.label_to_index[label]
            else:
                raise ValueError(f"Label '{label}' not found in COMMON_OBJECTS")
        elif isinstance(label, int):
            if 0 <= label < self.channel_num:
                return label
            else:
                raise ValueError(f"Index {label} out of range [0, {self.channel_num})")
        else:
            raise TypeError("Label must be either a string or an integer")

    def _update_channel(self, channel_index, local_map):
        """
        更新语义地图
        """
        # 确保channel_index在有效范围内
        if channel_index < 0 or channel_index >= self.channel_num:
            raise ValueError("channel_index must be in the range [0, channel_num)")

        # 直接取local_map和当前语义通道的最大值
        self.map[channel_index] = np.maximum(self.map[channel_index], local_map)

        # 更新次数加1
        self.update_times[channel_index] += 1

    def process_frame(self, obj_point_cloud_list, score_list, index_list):
        # 检查 3 个 list 大小相等
        if len(obj_point_cloud_list) != len(score_list) or len(
            obj_point_cloud_list
        ) != len(index_list):
            raise ValueError("All lists must have the same length")

        # 检查 label list 中的 label 都在 np.arange(self.channel_num) 范围内
        if not np.all(np.isin(index_list, np.arange(self.channel_num))):
            raise ValueError("All labels must be in the range [0, channel_num)")

        # 对于每个 list 进行遍历
        for points_world, score, label in zip(
            obj_point_cloud_list, score_list, index_list
        ):
            if len(points_world) == 0:
                continue

            # 2. 将世界坐标的点云投影到 2 维地图坐标，获得 local_map
            map_coords = self._world_to_map_coords(points_world)
            valid_indices = (
                (map_coords[:, 0] >= 0)
                & (map_coords[:, 0] < self.map_size)
                & (map_coords[:, 1] >= 0)
                & (map_coords[:, 1] < self.map_size)
            )
            valid_map_coords = map_coords[valid_indices]

            # 3. 将当前 local_map 与对应通道的 map 进行融合
            local_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
            if len(valid_map_coords) > 0:
                local_map[valid_map_coords[:, 1], valid_map_coords[:, 0]] = score

            self._update_channel(label, local_map)

    def visualize_map(self, channel_index=None, tmat_camera2world=None):
        """
        可视化当前的全局地图

        Args:
            channel_index: 要可视化的通道索引。如果为 None，则显示所有通道的叠加
            tmat_camera2world: 从相机坐标系到世界坐标系的变换矩阵
        """
        if channel_index is not None and not isinstance(channel_index, int):
            channel_index = self._get_channel_index(channel_index)

        if channel_index is None:
            # 为每个通道赋予不同的颜色并叠加
            # 创建一个彩色图像 (BGR格式)
            color_map_display = np.zeros(
                (self.map_size, self.map_size, 3), dtype=np.float32
            )

            # 为每个通道应用颜色
            for i in range(min(self.channel_num, len(COLOR_MAP_FOR_OBJECT))):
                channel_data = self.map[i]
                if np.max(channel_data) > 0:
                    # 归一化通道数据
                    normalized_channel = channel_data / np.max(channel_data)
                    # 应用颜色
                    for c in range(3):  # B, G, R 通道
                        color_value = (
                            COLOR_MAP_FOR_OBJECT[i][2 - c] / 255.0
                        )  # 注意BGR到RGB的转换
                        color_map_display[:, :, c] = np.maximum(
                            color_map_display[:, :, c], normalized_channel * color_value
                        )

            # 转换为 0-255 范围的 uint8 类型
            map_display = (color_map_display * 255).astype(np.uint8)
            map_display = np.flipud(map_display)
            map_display = cv2.cvtColor(map_display, cv2.COLOR_RGB2BGR)
        else:
            # 确保channel_index在有效范围内
            if channel_index < 0 or channel_index >= self.channel_num:
                raise ValueError("channel_index must be in the range [0, channel_num)")

            map_display = self.map[channel_index]
            # 归一化到0-255范围以便显示
            if np.max(map_display) > 0:
                map_display = (map_display / np.max(map_display) * 255).astype(np.uint8)
            else:
                map_display = (map_display * 255).astype(np.uint8)

            # Convert to 3-channel BGR image for consistent handling with robot visualization
            map_display = np.flipud(map_display)
            map_display = cv2.cvtColor(map_display, cv2.COLOR_GRAY2BGR)

        if tmat_camera2world is not None:
            robot_position_world = tmat_camera2world @ np.array([0, 0, 0, 1])
            robot_position_world = robot_position_world[:3]

            point_ahead_world = tmat_camera2world @ np.array([0, 0, 1, 1])
            point_ahead_world = point_ahead_world[:3]

            # 计算机器人的朝向
            robot_direction = point_ahead_world - robot_position_world
            robot_orientation = np.arctan2(robot_direction[1], robot_direction[0])

            # 将世界坐标转换为地图坐标
            robot_map_x = int(
                (robot_position_world[0] - self.map_center_x) / self.resolution
                + self.map_size / 2
            )
            robot_map_y = int(
                (robot_position_world[1] - self.map_center_y) / self.resolution
                + self.map_size / 2
            )

            # 确保坐标在地图范围内
            if 0 <= robot_map_x < self.map_size and 0 <= robot_map_y < self.map_size:
                # 在地图上绘制一个绿色的圆圈表示机器人位置
                # 注意：由于使用了flipud，y坐标需要翻转
                cv2.circle(
                    map_display,
                    (robot_map_x, self.map_size - 1 - robot_map_y),
                    5,
                    (0, 255, 0),
                    -1,
                )

                # 绘制一个箭头表示机器人的朝向
                # 计算箭头的终点
                arrow_length = 20  # 箭头长度（像素）
                end_x = int(robot_map_x + arrow_length * np.cos(robot_orientation))
                end_y = int(
                    robot_map_y + arrow_length * np.sin(robot_orientation)
                )  # 注意y轴翻转

                # 确保终点坐标在地图范围内
                end_x = max(0, min(self.map_size - 1, end_x))
                end_y = max(0, min(self.map_size - 1, end_y))

                # 绘制箭头
                cv2.arrowedLine(
                    map_display,
                    (robot_map_x, self.map_size - 1 - robot_map_y),
                    (end_x, self.map_size - 1 - end_y),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                    0,
                    0.3,
                )

        # 显示地图，不阻塞
        cv2.imshow("Semantic Map", map_display)
        cv2.waitKey(1)
