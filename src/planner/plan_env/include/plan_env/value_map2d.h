#ifndef _VALUE_MAP_H_
#define _VALUE_MAP_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <vector>

#include <plan_env/sdf_map2d.h>
#include <nav_msgs/OccupancyGrid.h>

using Eigen::Vector2d;
using Eigen::Vector2i;
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
class SDFMap2D;

class ValueMap {
public:
  ValueMap(SDFMap2D* sdf_map, ros::NodeHandle& nh);
  ~ValueMap(){};

  void updateVlfmValueMap(const Vector2d& sensor_pos, const double& sensor_yaw,
      const vector<Vector2i>& free_grids, const double& itm_score);
  void updateCustomValueMap(const nav_msgs::OccupancyGridConstPtr& msg);
  double getValue(const Vector2d& pos);
  double getValue(const Vector2i& idx);
  double getConfidence(const Vector2d& pos);
  double getConfidence(const Vector2i& idx);

private:
  double getFovConfidence(
      const Vector2d& sensor_pos, const double& sensor_yaw, const Vector2d& pt_pos);
  double normalizeAngle(double angle);

  vector<double> vlfm_value_buffer_;  // Grid-based semantic value storage
  vector<double> vlfm_confidence_buffer_;  // Grid-based confidence storage for weighted fusion

  vector<double> custom_value_buffer_;     // Grid-based semantic value storage
  vector<double> custom_confidence_buffer_;  // Grid-based confidence storage for weighted fusion

  // Utils
  SDFMap2D* sdf_map_;
};

inline double ValueMap::normalizeAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

inline double ValueMap::getConfidence(const Vector2d& pos)
{
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getConfidence(idx);
}

inline double ValueMap::getConfidence(const Vector2i& idx)
{
  int adr = sdf_map_->toAddress(idx);
  return vlfm_confidence_buffer_[adr];
}

inline double ValueMap::getValue(const Vector2d& pos)
{
  Vector2i idx;
  sdf_map_->posToIndex(pos, idx);
  return getValue(idx);
}

inline double ValueMap::getValue(const Vector2i& idx)
{
  int adr = sdf_map_->toAddress(idx);
  return value_buffer_[adr];
}

}  // namespace apexnav_planner
#endif
