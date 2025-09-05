import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from value_map.multi_semmantic_map import MultiSemanticMap
from value_map.obstacle_map import ObstacleMap


class ValueMap:
    def __init__(self, map_size=1000, resolution=0.05):
        """
        初始化ValueMap

        Args:
            map_size: 地图大小
            resolution: 地图分辨率
        """
        self.map_size = map_size
        self.resolution = resolution

        # 初始化价值地图
        self.value_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)

        # ROS发布器
        self.value_map_pub = rospy.Publisher(
            "/modified/value_map", OccupancyGrid, queue_size=1
        )

    def update_value_map(
        self, obstacle_map: ObstacleMap, semantic_map: MultiSemanticMap
    ):
        """
        更新价值地图
        """
        # 获取障碍物地图和语义地图的数据
        obstacle_data = obstacle_map.obstacle_space  # 障碍物通道
        free_space_data = obstacle_map.free_space  # 可通行区域通道

        # 获取语义地图数据
        semantic_data = semantic_map.map

        # 实现从左上角到右下角的距离衰减价值地图
        # 创建坐标网格
        x_coords, y_coords = np.meshgrid(
            np.arange(self.map_size), np.arange(self.map_size)
        )

        # 计算每个点到左上角(0,0)的距离
        distances = np.sqrt(x_coords**2 + y_coords**2)

        # 归一化距离到[0,1]范围
        max_distance = np.sqrt(self.map_size**2 + self.map_size**2)
        normalized_distances = distances / max_distance

        # 创建衰减价值地图：左上角值最高(1.0)，右下角值最低(0.0)
        self.value_map = 1.0 - normalized_distances

        # 将障碍物区域设置为最低价值
        self.value_map[obstacle_data > 0] = -1.0

        # 发布价值地图
        self.publish_value_map()

    def publish_value_map(self):
        """
        发布价值地图为OccupancyGrid消息
        """
        # 创建OccupancyGrid消息
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "world"
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.map_size
        grid_msg.info.height = self.map_size

        # 设置地图原点（需要根据实际情况调整）
        grid_msg.info.origin.position.x = -self.map_size * self.resolution / 2
        grid_msg.info.origin.position.y = -self.map_size * self.resolution / 2
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # 这里需要根据实际价值范围进行归一化
        normalized_map = (
            (self.value_map - np.min(self.value_map))
            / (np.max(self.value_map) - np.min(self.value_map))
            * 100
        ).astype(np.int8)

        # 处理障碍物区域（保持为-1表示未知或不可通行）
        normalized_map[self.value_map == -1000] = -1

        grid_msg.data = normalized_map.flatten().tolist()

        # 发布消息
        self.value_map_pub.publish(grid_msg)

    def visualize_value_map(self):
        """
        可视化价值地图
        """
        import cv2

        # 创建可视化图像
        vis_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # 归一化价值地图用于可视化
        if np.max(self.value_map) != np.min(self.value_map):
            normalized = (
                (self.value_map - np.min(self.value_map))
                / (np.max(self.value_map) - np.min(self.value_map))
                * 255
            ).astype(np.uint8)
        else:
            normalized = np.zeros_like(self.value_map, dtype=np.uint8)

        # 根据价值设置颜色
        # 负价值用红色表示
        vis_map[self.value_map < 0] = [0, 0, 255]  # 红色
        # 正价值用绿色表示
        vis_map[self.value_map > 0] = [0, 255, 0]  # 绿色
        # 零价值用蓝色表示
        vis_map[self.value_map == 0] = [255, 0, 0]  # 蓝色

        # 添加价值强度
        vis_map[:, :, 1] = normalized  # 绿色通道表示价值强度

        # 翻转图像以匹配坐标系
        vis_map = np.flipud(vis_map)

        # 显示图像
        cv2.imshow("Value Map", vis_map)
        cv2.waitKey(1)
