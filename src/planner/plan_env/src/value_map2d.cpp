/**
 * @file value_map2d.cpp
 * @brief Implementation of semantic value mapping system with confidence-weighted ITM score fusion
 *
 * This file implements the ValueMap class which provides semantic value mapping capabilities
 * for autonomous navigation systems. The implementation focuses on confidence-weighted fusion
 * of ITM (Image-Text Matching) scores using field-of-view based confidence modeling.
 *
 * Reference paper "VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation"
 *
 * @author Zager-Zhang
 */

#include <plan_env/value_map2d.h>

namespace apexnav_planner {
ValueMap::ValueMap(SDFMap2D* sdf_map, ros::NodeHandle& nh)
{
  this->sdf_map_ = sdf_map;
  int voxel_num = sdf_map_->getVoxelNum();
  vlfm_value_buffer_ = vector<double>(voxel_num, 0.0);
  vlfm_confidence_buffer_ = vector<double>(voxel_num, 0.0);

  custom_value_buffer_ = vector<double>(voxel_num, 0.0);
  custom_confidence_buffer_ = vector<double>(voxel_num, 0.0);
}

void ValueMap::updateVlfmValueMap(const Vector2d& sensor_pos, const double& sensor_yaw,
    const vector<Vector2i>& free_grids, const double& itm_score)
{
  for (const auto& grid : free_grids) {
    Vector2d pos;
    sdf_map_->indexToPos(grid, pos);
    int adr = sdf_map_->toAddress(grid);

    // Calculate FOV-based confidence for current observation
    double now_confidence = getFovConfidence(sensor_pos, sensor_yaw, pos);
    double now_value = itm_score;

    // Retrieve existing confidence and value
    double last_confidence = vlfm_confidence_buffer_[adr];
    double last_value = vlfm_value_buffer_[adr];

    // Apply confidence-weighted fusion with quadratic confidence combination
    vlfm_confidence_buffer_[adr] =
        (now_confidence * now_confidence + last_confidence * last_confidence) /
        (now_confidence + last_confidence);
    vlfm_value_buffer_[adr] = (now_confidence * now_value + last_confidence * last_value) /
                         (now_confidence + last_confidence);
  }
}

void ValueMap::updateCustomValueMap(const nav_msgs::OccupancyGridConstPtr& msg)
{

  // 直接将 OccupancyGrid 数据复制到 custom_value_buffer_ 中
  int data_size = msg->data.size();
  int buffer_size = custom_value_buffer_.size();
  // ROS_INFO_THROTTLE(2, "msg CustomValueMap size: %d", data_size);
  // ROS_INFO_THROTTLE(2, "custom_value_buffer_ size: %d", buffer_size);

  // 确保数据大小匹配
  if (data_size <= buffer_size) {
    for (int i = 0; i < data_size; ++i) {
      ROS_INFO_THROTTLE(1, "msg->data[i]: %d", msg->data[i]);
      custom_value_buffer_[i] = static_cast<double>(msg->data[i]);
    }

    // 如果数据大小小于 buffer 大小，将剩余部分置零
    for (int i = data_size; i < buffer_size; ++i) {
      custom_value_buffer_[i] = 0.0;
    }
  }
  else {
    // 如果数据大小超过 buffer 大小，只复制前 buffer_size 个数据
    for (int i = 0; i < buffer_size; ++i) {
      custom_value_buffer_[i] = static_cast<double>(msg->data[i]);
    }
  }

  // 更新置信度缓冲区（这里简单地将所有置信度设置为1.0）
  for (int i = 0; i < buffer_size; ++i) {
    custom_confidence_buffer_[i] = 1.0;
  }
}

double ValueMap::getFovConfidence(
    const Vector2d& sensor_pos, const double& sensor_yaw, const Vector2d& pt_pos)
{
  // Calculate relative position vector from sensor to target point
  Vector2d rel_pos = pt_pos - sensor_pos;
  double angle_to_point = atan2(rel_pos(1), rel_pos(0));

  // Normalize angles to [-π, π] range for consistent angular arithmetic
  double normalized_sensor_yaw = normalizeAngle(sensor_yaw);
  double normalized_angle_to_point = normalizeAngle(angle_to_point);
  double relative_angle = normalizeAngle(normalized_angle_to_point - normalized_sensor_yaw);

  // Apply cosine-squared FOV confidence model
  // FOV angle: 79° total field of view (typical RGB camera)
  double fov_angle = 79.0 * M_PI / 180.0;
  double value = std::cos(relative_angle / (fov_angle / 2) * (M_PI / 2));
  return value * value;  // Square for stronger center weighting
}

}  // namespace apexnav_planner
