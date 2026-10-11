#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"

// 移植说明：sp_vision 原版的 tracker.hpp 还带一个"全向感知"重载
// （track(std::vector<omniperception::DetectionResult>, ...)），它把
// decider/detection/usbcamera/thread_pool 一整串依赖拖进来。我们只验证
// "单主相机 + 局部拟合器"这条路，所以把那一路裁掉了 —— 滤波器/关联/状态机
// 本身一行没动。

namespace spv
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  /// 移植扩展（**非 sp_vision 原版**）：`track()` 返回的是内部 target_ 的**拷贝**，
  /// 而局部拟合器注入一路观测必须拿到内部那个。只暴露引用，关联/状态机逻辑没动。
  Target & target_mutable() { return target_; }

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

private:
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
};

}  // namespace spv

#endif  // AUTO_AIM__TRACKER_HPP
