#ifndef _MULTI_CHANNEL_OBJECT_MAP2D_H_
#define _MULTI_CHANNEL_OBJECT_MAP2D_H_

// ROS and system includes
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <nav_msgs/OccupancyGrid.h>

// Internal mapping components
#include <plan_env/sdf_map2d.h>

// PCL for point cloud processing
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

namespace apexnav_planner {
class SDFMap2D;

struct MultiChannelDetectedObject {
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud;
  double score;  // 置信度分数
  int label;     // 通道标签
};

class MultiChannelObjectMap2D {
public:
  MultiChannelObjectMap2D(SDFMap2D* sdf_map, ros::NodeHandle& nh);
  ~MultiChannelObjectMap2D() = default;

  // 处理检测到的对象，需要传感器位姿信息
  void processDetectedObjects(const std::vector<MultiChannelDetectedObject>& detected_objects,
      const Eigen::Vector3d& sensor_pos, const Eigen::Quaterniond& sensor_orient);

  // 发布语义地图
  void publishSemanticMaps();

  // 设置置信度阈值
  void setConfidenceThreshold(double val)
  {
    min_confidence_ = val;
  }

private:
  // 点云坐标转换
  void transformPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, const Eigen::Vector3d& sensor_pos,
      const Eigen::Quaterniond& sensor_orient);

  // 更新通道置信度地图
  void updateChannelMap(
      int channel, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double confidence);

  // 世界坐标转网格坐标
  bool worldToGrid(const Eigen::Vector2d& world_pos, Eigen::Vector2i& grid_idx);

  // 核心数据
  SDFMap2D* sdf_map_;
  ros::NodeHandle nh_;
  int num_channels_;

  // 地图参数
  double resolution_;
  Eigen::Vector2d origin_;
  int grid_width_;
  int grid_height_;

  // 置信度参数
  double min_confidence_;
  double update_rate_;

  // 通道置信度地图 (行:高度, 列:宽度)
  std::vector<Eigen::MatrixXd> channel_maps_;

  // ROS发布器
  std::vector<ros::Publisher> semantic_map_pubs_;
};

}  // namespace apexnav_planner

#endif  // _MULTI_CHANNEL_OBJECT_MAP2D_H_
