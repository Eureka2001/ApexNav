#include <plan_env/multi_channel_object_map2d.h>
#include <Eigen/Geometry>
#include <nav_msgs/OccupancyGrid.h>

namespace apexnav_planner {

MultiChannelObjectMap2D::MultiChannelObjectMap2D(
    SDFMap2D* sdf_map, ros::NodeHandle& nh)
{
  // 初始化核心参数
  this->sdf_map_ = sdf_map;
  int num_channels = 20;
  this->num_channels_ = num_channels;

  resolution_ = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);
  origin_ = origin;
  grid_width_ = static_cast<int>(size.x() / resolution_);
  grid_height_ = static_cast<int>(size.y() / resolution_);

  ROS_INFO_STREAM("MultiChannelObjectMap2D initialized with origin: ["
                  << origin_.x() << ", " << origin_.y() << "] resolution: " << resolution_);

  // 初始化通道置信度地图
  channel_maps_.resize(num_channels_);
  for (int i = 0; i < num_channels_; ++i) {
    channel_maps_[i] = Eigen::MatrixXd::Zero(grid_height_, grid_width_);
  }

  // 设置ROS发布器
  semantic_map_pubs_.resize(num_channels_);
  for (int i = 0; i < num_channels_; ++i) {
    std::string topic_name = "/multi_channel_object/semantic_map_" + std::to_string(i);
    semantic_map_pubs_[i] = nh.advertise<nav_msgs::OccupancyGrid>(topic_name, 10);
  }

  // 加载配置参数
  nh.param("multi_channel_object/min_confidence", min_confidence_, 0.5);
  nh.param("multi_channel_object/update_rate", update_rate_, 10.0);
}

void MultiChannelObjectMap2D::processDetectedObjects(
    const std::vector<MultiChannelDetectedObject>& detected_objects,
    const Eigen::Vector3d& sensor_pos, const Eigen::Quaterniond& sensor_orient)
{
  for (const auto& obj : detected_objects) {
    int channel = obj.label;
    if (channel < 0 || channel >= num_channels_) {
      ROS_WARN("Invalid channel ID: %d", channel);
      continue;
    }

    // 转换点云到世界坐标系
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    transformPointCloud(obj.cloud, transformed_cloud, sensor_pos, sensor_orient);

    // 更新通道置信度地图
    updateChannelMap(channel, transformed_cloud, obj.score);
  }
}

void MultiChannelObjectMap2D::transformPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud, const Eigen::Vector3d& sensor_pos,
    const Eigen::Quaterniond& sensor_orient)
{
  output_cloud->resize(input_cloud->size());
  Eigen::Matrix3d rotation = sensor_orient.toRotationMatrix();

  for (size_t i = 0; i < input_cloud->size(); ++i) {
    const auto& pt = input_cloud->points[i];
    Eigen::Vector3d point(pt.x, pt.y, pt.z);
    Eigen::Vector3d transformed = rotation * point + sensor_pos;
    output_cloud->points[i].x = transformed.x();
    output_cloud->points[i].y = transformed.y();
    output_cloud->points[i].z = transformed.z();
  }
}

void MultiChannelObjectMap2D::updateChannelMap(
    int channel, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double confidence)
{
  for (const auto& pt : cloud->points) {
    Eigen::Vector2d grid_pos(pt.x, pt.y);
    Eigen::Vector2i grid_idx;

    if (worldToGrid(grid_pos, grid_idx)) {
      // 更新置信度，取当前值和新值的最大值
      if (confidence > channel_maps_[channel](grid_idx.y(), grid_idx.x())) {
        channel_maps_[channel](grid_idx.y(), grid_idx.x()) = confidence;
      }
    }
  }
}

bool MultiChannelObjectMap2D::worldToGrid(
    const Eigen::Vector2d& world_pos, Eigen::Vector2i& grid_idx)
{
  grid_idx.x() = static_cast<int>((world_pos.x() - origin_.x()) / resolution_);
  grid_idx.y() = static_cast<int>((world_pos.y() - origin_.y()) / resolution_);

  if (grid_idx.x() < 0 || grid_idx.x() >= grid_width_ || grid_idx.y() < 0 ||
      grid_idx.y() >= grid_height_) {
    ROS_WARN_THROTTLE(1.0, "Point (%.2f, %.2f) outside map bounds", world_pos.x(), world_pos.y());
    return false;
  }
  return true;
}

void MultiChannelObjectMap2D::publishSemanticMaps()
{
  for (int channel = 0; channel < num_channels_; ++channel) {
    nav_msgs::OccupancyGrid map_msg;
    map_msg.header.frame_id = "world";
    map_msg.header.stamp = ros::Time::now();
    map_msg.info.resolution = resolution_;
    map_msg.info.origin.position.x = origin_.x();
    map_msg.info.origin.position.y = origin_.y();
    map_msg.info.origin.orientation.w = 1.0;
    map_msg.info.width = grid_width_;
    map_msg.info.height = grid_height_;

    // 转换置信度值到0-100范围
    map_msg.data.resize(grid_width_ * grid_height_);
    for (int y = 0; y < grid_height_; ++y) {
      for (int x = 0; x < grid_width_; ++x) {
        double conf = channel_maps_[channel](y, x);
        int value = static_cast<int>(conf * 100.0);
        map_msg.data[y * grid_width_ + x] = std::min(100, std::max(0, value));
      }
    }

    semantic_map_pubs_[channel].publish(map_msg);
  }
}

}  // namespace apexnav_planner
