import threading

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


class ObstacleMap:
    """障碍物地图类，用于处理和可视化障碍物和可通行区域"""

    def __init__(self):
        """初始化障碍物地图"""
        self.map_size = 1000
        self.resolution = 0.05  # 地图分辨率 (m/cell)
        self.origin_x = -25  # 地图原点x坐标 (m)
        self.origin_y = -25  # 地图原点y坐标 (m)
        self.map_width = 20.0  # 地图宽度 (m)
        self.map_height = 20.0  # 地图高度 (m)
        self.limits = [self.origin_x, self.origin_x + self.map_width]  # 地图边界

        # 初始化双通道地图
        # 通道0: 可通行区域 (free space)
        # 通道1: 障碍物区域 (occupied space)
        self.channel_num = 2
        self.map = np.zeros(
            (self.channel_num, self.map_size, self.map_size), dtype=np.float32
        )
        self.grid_lock = threading.Lock()

        # 订阅障碍物和可通行区域的点云数据
        rospy.Subscriber("/grid_map/occupied", PointCloud2, self.occupied_callback)
        rospy.Subscriber("/grid_map/free", PointCloud2, self.free_callback)

        # 发布OccupancyGrid消息用于RViz可视化
        self.grid_pub = rospy.Publisher(
            "/visualization/obstacle_map", OccupancyGrid, queue_size=1
        )

        # OpenCV可视化相关
        self.window_name = "Obstacle Map"
        self.display_image = None

    # 添加属性访问器，允许外部像访问变量一样访问两个通道
    @property
    def free_space(self):
        """返回可通行区域通道（通道0）"""
        return self.map[0]

    @property
    def obstacle_space(self):
        """返回障碍物区域通道（通道1）"""
        return self.map[1]

    @property
    def channels(self):
        """返回所有通道"""
        return self.map

    def reset(self):
        """重置地图，将所有通道清零"""
        with self.grid_lock:
            self.map.fill(0.0)

    def _update_channel(self, channel_index, points):
        """
        更新指定通道的地图

        Args:
            channel_index (int): 通道索引 (0: free, 1: occupied)
            points (np.ndarray): 点云数据 (N x 3)
        """
        # 确保channel_index在有效范围内
        if channel_index < 0 or channel_index >= self.channel_num:
            raise ValueError("channel_index must be in the range [0, channel_num)")

        if len(points) == 0:
            return

        # 提取xy坐标
        xy_points = points[:, :2]

        # 创建局部地图
        local_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)

        # 将点云坐标转换为地图索引
        indices = (
            (xy_points - [self.origin_x, self.origin_y]) / self.resolution
        ).astype(np.int32)
        indices = np.clip(indices, 0, self.map_size - 1)

        # 在局部地图中标记点云位置
        local_map[indices[:, 1], indices[:, 0]] = 1.0

        # 更新地图通道（直接替换而不是累加）
        self.map[channel_index] = local_map

    def occupied_callback(self, msg):
        """
        处理障碍物点云数据的回调函数

        Args:
            msg (PointCloud2): 障碍物点云数据
        """
        points = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        )
        with self.grid_lock:
            self._update_channel(1, points)  # 通道1用于障碍物

        # 发布OccupancyGrid消息
        self.publish_occupancy_grid()

    def free_callback(self, msg):
        """
        处理可通行区域点云数据的回调函数

        Args:
            msg (PointCloud2): 可通行区域点云数据
        """
        points = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        )
        with self.grid_lock:
            self._update_channel(0, points)  # 通道0用于可通行区域

    def publish_occupancy_grid(self):
        """发布ROS标准地图消息用于RViz叠加"""
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "world"  # 统一坐标系
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.map_size
        grid_msg.info.height = self.map_size

        # 设置地图原点
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # 合并两个通道的信息
        # 障碍物优先级最高，然后是可通行区域，最后是未知区域
        combined_grid = np.full(
            (self.map_size, self.map_size), -1, dtype=np.int8
        )  # 默认未知
        combined_grid[self.map[0] > 0] = 0  # 可通行区域标记为0
        combined_grid[self.map[1] > 0] = 100  # 障碍物标记为100

        grid_msg.data = combined_grid.flatten().tolist()
        self.grid_pub.publish(grid_msg)

    def _generate_display_image(self):
        """生成用于显示的地图图像"""
        # 获取当前地图数据
        with self.grid_lock:
            map_data = self.map.copy()

        # 合并两个通道用于可视化
        # 使用不同的数值来表示不同的区域
        combined_grid = np.full(
            (self.map_size, self.map_size), 0, dtype=np.float32
        )  # 默认未知区域
        combined_grid[map_data[0] > 0] = 1  # 可通行区域
        combined_grid[map_data[1] > 0] = 2  # 障碍物

        # 创建彩色图像
        # 未知区域: 黑色 (0, 0, 0)
        # 可通行区域: 绿色 (0, 255, 0)
        # 障碍物: 红色 (0, 0, 255)
        display_image = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        display_image[combined_grid == 1] = [0, 255, 0]  # 绿色
        display_image[combined_grid == 2] = [0, 0, 255]  # 红色

        # 翻转图像以匹配坐标系
        display_image = np.flipud(display_image)

        return display_image

    def visualize(self):
        """按需可视化地图"""
        # 创建OpenCV窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # 生成显示图像
        self.display_image = self._generate_display_image()

        # 显示图像
        cv2.imshow(self.window_name, self.display_image)
        cv2.waitKey(1)  # 短暂等待以确保窗口更新

        return self.window_name

    def update_visualization(self):
        """更新可视化"""
        # 如果还没有创建图像，直接返回
        if self.display_image is None:
            return

        # 生成显示图像
        self.display_image = self._generate_display_image()

        # 显示图像
        cv2.imshow(self.window_name, self.display_image)
        cv2.waitKey(1)  # 短暂等待以确保窗口更新


if __name__ == "__main__":
    try:
        rospy.init_node("obstacle_map", anonymous=True)
        obstacle_map = ObstacleMap()
        rate = rospy.Rate(20)  # 提高更新频率到20Hz

        # 可视化地图
        # window_name = obstacle_map.visualize()

        # 持续更新地图
        while not rospy.is_shutdown():
            # obstacle_map.update_visualization()
            rate.sleep()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
