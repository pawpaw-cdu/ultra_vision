// 因子图离线原型（第 1 步）：Ceres 滑动窗口 + "每帧独立的板位姿"
//
// 要回答的问题（来自实车 A/B 的实测）：
//   UV/像素观测塞进 EKF 时**没法两全** —— 增益给到能改善重投影（板心 p50 4.3px）
//   时，静止段位置抖动 ×6、yaw 抖动 ×7、ω 冒出 +1.56 rad/s 的假偏置、丢帧 +9%；
//   阻尼到不伤状态时重投影与基线打平。根因是**模型误差跨帧相关**，而 EKF 里没有
//   "每帧一个独立变量"去吸收它。
//
// 本原型就是加那个变量（对齐 jlu_vision_26 的因子结构）：
//   变量：每帧底盘 {C_k(3), V_k(3), θ_k, ω_k} + 每帧每块**被检板**的 {P(3), ψ}
//   因子：① 运动（C_{k+1}=C_k+V_k·dt、V 常值；θ/ω 同理）
//         ② 观测（每个角点一个 2 维重投影因子，σ=1px，含畸变）
//         ③ "车心→板心"分解成切向/径向（**半径只由径向项约束**）+ z + 相位
//            （对齐 jlu 的 ArmorRadiusCenterZFactor：避免 yaw 相位误差通过切向
//             残差把半径拉小）
// 几何 r **固定**为实测值：本车 0.29（两种独立方法一致）；仿真车要显式给 0.2009。
//
// 用法：
//   ./fgo_probe --detections /tmp/det2.csv \
//       --camera configs/camera.yaml --extrinsics configs/spv_extrinsics.yaml \
//       --window 12 --radius 0.29 --log /tmp/fgo.csv --detlog /tmp/fgo_det.csv
//
// 落盘格式与 sp_vision 回放器（tools/local_fit_replay.cpp）逐列一致，
// 这样同一套 python 度量可以直接对比（静止段抖动 / 跨板重投影 / 丢帧率）。

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
  constexpr double kPi = 3.14159265358979323846;
  // 与 sp_vision Solver::reproject_armor / SMALL_ARMOR_POINTS 一致
  constexpr double kPlateW = 0.135;       // 小装甲长（官方规范）
  constexpr double kLightbarLen = 0.056;  // 灯条上下端距离（PnP 用的同一组点）
  constexpr double kPlatePitch = 15.0 * kPi / 180.0;

  struct Cam
  {
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;

    Eigen::Vector2d project(const Eigen::Vector3d & pc) const
    {
      const double z = std::max(1e-3, pc.z());
      const double x = pc.x() / z, y = pc.y() / z;
      const double r2 = x * x + y * y;
      const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
      const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
      const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
      return {fx * xd + cx, fy * yd + cy};
    }
  };

  /// 板坐标系→世界：与 sp_vision Solver::reproject_armor 的 R_armor2world 逐位一致
  Eigen::Matrix3d armor_R(double yaw)
  {
    const double sy = std::sin(yaw), cy = std::cos(yaw);
    const double sp = std::sin(kPlatePitch), cp = std::cos(kPlatePitch);
    Eigen::Matrix3d R;
    R << cy * cp, -sy, cy * sp, sy * cp, cy, sy * sp, -sp, 0.0, cp;
    return R;
  }

  /// 板局部坐标下的 4 个角点，顺序与 sp_vision `Armor::points` = [左上,右上,右下,左下] 一致
  std::array<Eigen::Vector3d, 4> object_points()
  {
    const double a = kPlateW * 0.5, b = kLightbarLen * 0.5;
    return {Eigen::Vector3d(0, a, b), Eigen::Vector3d(0, -a, b), Eigen::Vector3d(0, -a, -b),
            Eigen::Vector3d(0, a, -b)};
  }

  // ---- 因子 -------------------------------------------------------------------
  struct MotionCost
  {
    MotionCost(double dt, double sigma) : dt_(dt), sigma_(sigma) {}
    template <typename T>
    bool operator()(const T * const c0, const T * const v0, const T * const c1, T * res) const
    {
      for (int i = 0; i < 3; ++i) res[i] = (c1[i] - c0[i] - v0[i] * T(dt_)) / T(sigma_);
      return true;
    }
    double dt_, sigma_;
  };
  struct VelCost
  {
    explicit VelCost(double sigma) : sigma_(sigma) {}
    template <typename T>
    bool operator()(const T * const v0, const T * const v1, T * res) const
    {
      for (int i = 0; i < 3; ++i) res[i] = (v1[i] - v0[i]) / T(sigma_);
      return true;
    }
    double sigma_;
  };
  struct YawMotionCost
  {
    YawMotionCost(double dt, double sigma) : dt_(dt), sigma_(sigma) {}
    template <typename T>
    bool operator()(const T * const t0, const T * const w0, const T * const t1, T * res) const
    {
      res[0] = (t1[0] - t0[0] - w0[0] * T(dt_)) / T(sigma_);
      return true;
    }
    double dt_, sigma_;
  };
  struct WCost
  {
    explicit WCost(double sigma) : sigma_(sigma) {}
    template <typename T>
    bool operator()(const T * const w0, const T * const w1, T * res) const
    {
      res[0] = (w1[0] - w0[0]) / T(sigma_);
      return true;
    }
    double sigma_;
  };
  /// "车心→板心"分解：切向 / 径向(含半径) / z / 相位
  struct TieCost
  {
    TieCost(double radius, int plate_index, int armor_num, const double * sig)
    : radius_(radius), idx_(plate_index), n_(armor_num)
    {
      for (int i = 0; i < 4; ++i) sig_[i] = sig[i];
    }
    /// 侧板（id 1/3）的高度偏移（= sp_vision 状态的 x[10]）
    void set_h(double h) { h_ = (idx_ == 1 || idx_ == 3) ? h : 0.0; }
    template <typename T>
    bool operator()(
      const T * const C, const T * const th, const T * const P, const T * const psi, T * res) const
    {
      const T a = th[0] + T(idx_ * 2.0 * kPi / n_);
      const T nx = -cos(a), ny = -sin(a);  // 车心→板心 = 板的外法向
      const T tx = -ny, ty = nx;
      const T dx = P[0] - C[0], dy = P[1] - C[1];
      res[0] = (tx * dx + ty * dy) / T(sig_[0]);
      res[1] = (nx * dx + ny * dy - T(radius_)) / T(sig_[1]);
      res[2] = (P[2] - C[2] - T(h_)) / T(sig_[2]);   // h_ = 侧板的高度偏移（id 1/3）
      res[3] = (psi[0] - a) / T(sig_[3]);
      return true;
    }
    double radius_, sig_[4];
    double h_ = 0.0;
    int idx_, n_;
  };
  /// 一个角点的 2 维重投影因子
  struct ReprojCost
  {
    ReprojCost(
      const Cam & cam, const Eigen::Matrix3d & R_w2c, const Eigen::Vector3d & cam0,
      const Eigen::Vector3d & obj, const Eigen::Vector2d & px, double sigma)
    : cam_(cam), R_(R_w2c), c0_(cam0), obj_(obj), px_(px), sigma_(sigma)
    {
    }
    template <typename T>
    bool operator()(const T * const P, const T * const psi, T * res) const
    {
      const T y = psi[0];
      const T sy = sin(y), cy = cos(y);
      const T sp = T(std::sin(kPlatePitch)), cp = T(std::cos(kPlatePitch));
      const T ox = T(obj_.x()), oy = T(obj_.y()), oz = T(obj_.z());
      const T wx = P[0] + (cy * cp * ox - sy * oy + cy * sp * oz);
      const T wy = P[1] + (sy * cp * ox + cy * oy + sy * sp * oz);
      const T wz = P[2] + (-sp * ox + cp * oz);
      const T dx = wx - T(c0_.x()), dy = wy - T(c0_.y()), dz = wz - T(c0_.z());
      const T xc = T(R_(0, 0)) * dx + T(R_(0, 1)) * dy + T(R_(0, 2)) * dz;
      const T yc = T(R_(1, 0)) * dx + T(R_(1, 1)) * dy + T(R_(1, 2)) * dz;
      const T zc = T(R_(2, 0)) * dx + T(R_(2, 1)) * dy + T(R_(2, 2)) * dz;
      const T zz = zc < T(1e-3) ? T(1e-3) : zc;
      const T x = xc / zz, yy = yc / zz;
      const T r2 = x * x + yy * yy;
      const T radial =
        T(1.0) + T(cam_.k1) * r2 + T(cam_.k2) * r2 * r2 + T(cam_.k3) * r2 * r2 * r2;
      const T xd = x * radial + T(2.0 * cam_.p1) * x * yy + T(cam_.p2) * (r2 + T(2.0) * x * x);
      const T yd =
        yy * radial + T(cam_.p1) * (r2 + T(2.0) * yy * yy) + T(2.0 * cam_.p2) * x * yy;
      res[0] = (T(cam_.fx) * xd + T(cam_.cx) - T(px_.x())) / T(sigma_);
      res[1] = (T(cam_.fy) * yd + T(cam_.cy) - T(px_.y())) / T(sigma_);
      return true;
    }
    Cam cam_;
    Eigen::Matrix3d R_;
    Eigen::Vector3d c0_, obj_;
    Eigen::Vector2d px_;
    double sigma_;
  };
  struct Prior3Cost
  {
    Prior3Cost(const double * target, double sigma) : sigma_(sigma)
    {
      for (int i = 0; i < 3; ++i) t_[i] = target[i];
    }
    template <typename T>
    bool operator()(const T * const x, T * res) const
    {
      for (int i = 0; i < 3; ++i) res[i] = (x[i] - T(t_[i])) / T(sigma_);
      return true;
    }
    double t_[3], sigma_;
  };
  struct Prior1Cost
  {
    Prior1Cost(double target, double sigma) : t_(target), sigma_(sigma) {}
    template <typename T>
    bool operator()(const T * const x, T * res) const
    {
      res[0] = (x[0] - T(t_)) / T(sigma_);
      return true;
    }
    double t_, sigma_;
  };
}  // namespace

namespace
{
  struct Detection
  {
    Eigen::Vector2d p[4];   // 顺序 = sp_vision `Armor::points` = [左上,右上,右下,左下]
  };
  struct Frame
  {
    long long tu = 0;
    double t = 0.0;
    std::vector<Detection> dets;
    // 变量（warm start 就地保存）
    Eigen::Vector3d C = Eigen::Vector3d::Zero(), V = Eigen::Vector3d::Zero();
    double th = 0.0, w = 0.0;
    std::vector<int> plate_id;
    std::vector<Eigen::Vector3d> P;
    std::vector<double> psi;
    Eigen::Matrix3d R_w2c = Eigen::Matrix3d::Identity();
    Eigen::Vector3d cam0 = Eigen::Vector3d::Zero();
  };

  std::vector<std::string> splitCsv(const std::string & line)
  {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
  }
  std::string trim(const std::string & s)
  {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
  }
  bool num(const std::vector<std::string> & r, int i, double & v)
  {
    if (i < 0 || i >= static_cast<int>(r.size())) return false;
    const std::string s = trim(r[static_cast<std::size_t>(i)]);
    if (s.empty() || s == "nan") return false;
    try { v = std::stod(s); } catch (...) { return false; }
    return std::isfinite(v);
  }
  std::vector<double> readVec(const YAML::Node & n)
  {
    if (!n) return {};
    if (n["data"]) return n["data"].as<std::vector<double>>();
    return n.as<std::vector<double>>();
  }
  /// Ultra_Vision 的相机→车体(base)旋转（与回放器/实车同一份约定）
  Eigen::Matrix3d cameraToBaseRotation(double yaw, double pitch)
  {
    const double cy = std::cos(yaw), sy = std::sin(yaw);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    Eigen::Matrix3d bevy;
    bevy << cy, -sy * sp, -sy * cp, 0.0, cp, -sp, sy, cy * sp, cy * cp;
    const Eigen::Vector3d s(1.0, -1.0, -1.0);
    return s.asDiagonal() * bevy * s.asDiagonal();
  }
}  // namespace

int main(int argc, char ** argv)
{
  std::string det_path, cam_path = "configs/camera.yaml";
  std::string ext_path = "configs/spv_extrinsics.yaml";
  std::string log_path, detlog_path;
  int window = 12;
  double radius = 0.29, sigma_px = 1.0, yaw_sign = -1.0, pitch_sign = -1.0;
  // 侧板（id 1/3）相对前后板（id 0/2）的**高度偏移** —— 就是 sp_vision 状态里的 x[10]。
  // ⚠ 实测这台车的两块同帧可见板 z 差 ≈0.10 m；模型假设四块等高(h=0)时，
  //   双检测帧的最优拟合会剩 ~100px/板（0.10m @1.3m ≈ 92px）—— 这就是"四块板高度不统一"。
  double h_offset = 0.0;
  // 双板构造残差门（px，每板）：超过它就**不认这一对**，退回"只喂一块"。
  // 这就是"双板残差大就丢弃"的量化版本；默认关(=inf)保持旧行为。
  double pair_gate_px = std::numeric_limits<double>::infinity();
  // 锚板锁定迟滞（对齐 Ultra_Vision tracker 的 plate_lock_margin: 1.35）
  double plate_lock_margin = 1.35;
  bool single_only = false;   // 只用"最近的那一块"检测（诊断双检测的影响）
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    if (a == "--detections" && i + 1 < argc) det_path = argv[++i];
    else if (a == "--camera" && i + 1 < argc) cam_path = argv[++i];
    else if (a == "--extrinsics" && i + 1 < argc) ext_path = argv[++i];
    else if (a == "--window" && i + 1 < argc) window = std::atoi(argv[++i]);
    else if (a == "--radius" && i + 1 < argc) radius = std::atof(argv[++i]);
    else if (a == "--h" && i + 1 < argc) h_offset = std::atof(argv[++i]);
    else if (a == "--pair-gate" && i + 1 < argc) pair_gate_px = std::atof(argv[++i]);
    else if (a == "--lock-margin" && i + 1 < argc) plate_lock_margin = std::atof(argv[++i]);
    else if (a == "--sigma-px" && i + 1 < argc) sigma_px = std::atof(argv[++i]);
    else if (a == "--yaw-sign" && i + 1 < argc) yaw_sign = std::atof(argv[++i]);
    else if (a == "--pitch-sign" && i + 1 < argc) pitch_sign = std::atof(argv[++i]);
    else if (a == "--log" && i + 1 < argc) log_path = argv[++i];
    else if (a == "--detlog" && i + 1 < argc) detlog_path = argv[++i];
    else if (a == "--single") single_only = true;
    else { std::cerr << "未知参数 " << a << std::endl; return 2; }
  }
  if (det_path.empty())
  {
    std::cerr << "用法: fgo_probe --detections <csv> [--window 12] [--radius 0.29]\n";
    return 2;
  }

  Cam cam;
  cv::Mat camK, camD;
  {
    const YAML::Node root = YAML::LoadFile(cam_path);
    const YAML::Node sec = root["hikcamera"] ? root["hikcamera"] : root["galaxy"];
    const auto K = readVec(sec["camera_matrix"]);
    const auto D = readVec(sec["dist_coeffs"] ? sec["dist_coeffs"] : sec["distort_coeffs"]);
    cam.fx = K[0]; cam.fy = K[4]; cam.cx = K[2]; cam.cy = K[5];
    if (D.size() >= 5) { cam.k1 = D[0]; cam.k2 = D[1]; cam.p1 = D[2]; cam.p2 = D[3]; cam.k3 = D[4]; }
    camK = (cv::Mat_<double>(3, 3) << cam.fx, 0.0, cam.cx, 0.0, cam.fy, cam.cy, 0.0, 0.0, 1.0);
    camD = (cv::Mat_<double>(1, 5) << cam.k1, cam.k2, cam.p1, cam.p2, cam.k3);
    std::printf(
      "[fgo] 内参 fx=%.2f fy=%.2f cx=%.2f cy=%.2f  D=(%.5f %.5f %.6f %.6f %.5f)\n", cam.fx,
      cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, cam.p1, cam.p2, cam.k3);
  }
  Eigen::Matrix3d R_c2g = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_c2g = Eigen::Vector3d::Zero();
  {
    const YAML::Node e = YAML::LoadFile(ext_path);
    const auto a = e["R_gimbal2imubody"].as<std::vector<double>>();
    const auto b = e["R_camera2gimbal"].as<std::vector<double>>();
    const auto c = e["t_camera2gimbal"].as<std::vector<double>>();
    (void)a;
    R_c2g = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(b.data());
    t_c2g << c[0], c[1], c[2];
    std::printf(
      "[fgo] 外参 <- %s  t_camera2gimbal=(%.3f %.3f %.3f)\n", ext_path.c_str(), t_c2g.x(),
      t_c2g.y(), t_c2g.z());
  }

  // ---- 读检测流：按 time_us 排序 + 同帧合并（与修好的回放器同口径）---------------
  std::vector<Frame> frames;
  {
    std::ifstream in(det_path);
    if (!in) { std::cerr << "打不开 " << det_path << "\n"; return 1; }
    std::string line;
    std::getline(in, line);
    const auto head = splitCsv(line);
    int i_t = -1, i_y = -1, i_p = -1, cu[4] = {-1, -1, -1, -1}, cv[4] = {-1, -1, -1, -1};
    for (std::size_t i = 0; i < head.size(); ++i)
    {
      const std::string h = trim(head[i]);
      if (h == "time_us") i_t = static_cast<int>(i);
      else if (h == "cmd_yaw") i_y = static_cast<int>(i);
      else if (h == "cmd_pitch") i_p = static_cast<int>(i);
      else if (h.size() == 2 && h[0] == 'u') cu[h[1] - '0'] = static_cast<int>(i);
      else if (h.size() == 2 && h[0] == 'v') cv[h[1] - '0'] = static_cast<int>(i);
    }
    std::vector<std::pair<long long, std::string>> rec;
    while (std::getline(in, line))
    {
      if (line.empty()) continue;
      const auto r = splitCsv(line);
      double tu = 0.0;
      if (!num(r, i_t, tu)) continue;
      rec.emplace_back(static_cast<long long>(std::llround(tu)), line);
    }
    std::stable_sort(
      rec.begin(), rec.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    const long long t0 = rec.empty() ? 0 : rec.front().first;
    for (std::size_t i = 0; i < rec.size();)
    {
      const long long tu = rec[i].first;
      Frame f;
      f.tu = tu;
      f.t = static_cast<double>(tu - t0) * 1e-6;
      double gy = 0.0, gp = 0.0;
      while (i < rec.size() && rec[i].first == tu)
      {
        const auto r = splitCsv(rec[i].second);
        ++i;
        double y = 0.0, p = 0.0;
        if (num(r, i_y, y)) gy = y;
        if (num(r, i_p, p)) gp = p;
        double u = 0.0, v = 0.0;
        if (!(num(r, cu[0], u) && num(r, cv[0], v))) continue;
        Detection d;
        bool ok = true;
        for (int k = 0; k < 4; ++k)
        {
          double uk = 0.0, vk = 0.0;
          if (!num(r, cu[k], uk) || !num(r, cv[k], vk)) { ok = false; break; }
          // CSV 顺序 = [左下,左上,右上,右下] → points = [左上,右上,右下,左下] = {1,2,3,0}
          const int src = (k + 1) % 4;
          double su = 0.0, sv = 0.0;
          num(r, cu[src], su);
          num(r, cv[src], sv);
          d.p[k] = Eigen::Vector2d(su, sv);
        }
        if (ok) f.dets.push_back(d);
      }
      const Eigen::Matrix3d R_g2w = cameraToBaseRotation(yaw_sign * gy, pitch_sign * gp);
      f.R_w2c = (R_g2w * R_c2g).transpose();
      f.cam0 = R_g2w * t_c2g;
      frames.push_back(f);
    }
    std::printf(
      "[fgo] 记录 %zu 行 → %zu 帧（按 time_us 排序 + 同帧合并）\n", rec.size(), frames.size());
  }

  std::ofstream lg, dl;
  if (!log_path.empty())
  {
    lg.open(log_path, std::ios::out | std::ios::trunc);
    lg << "t,state,n_det,a,w,x,z,uv_updates,det_u,det_v,m0u,m0v,m1u,m1v,m2u,m2v,m3u,m3v,"
          "z4,r8,l9,h10\n";
  }
  if (!detlog_path.empty())
  {
    dl.open(detlog_path, std::ios::out | std::ios::trunc);
    dl << "t,fidx,ndet,di,det_u,det_v,plate,center_err,corner_rmse,w,x,y,yaw,"
          "px,py,pz,yaw_pnp,px_g,py_g,pz_g,w135_px,h55_px\n";
  }

  const auto obj = object_points();
  std::deque<Frame> win;
  bool inited = false;
  Eigen::Vector3d last_C = Eigen::Vector3d::Zero(), last_V = Eigen::Vector3d::Zero();
  double last_th = 0.0, last_w = 0.0;
  int anchor_id_ = -1;   // 上一帧锚定的板号（选板迟滞用）

  auto plate_center = [&](const Eigen::Vector3d & C, double th, int id) {
    const double a = th + id * 2.0 * kPi / 4.0;
    // 与 sp_vision 的 h_armor_xyz 一致：侧板（id 1/3）的 z 比前后板高 h_offset
    const double z = C.z() + ((id == 1 || id == 3) ? h_offset : 0.0);
    return Eigen::Vector3d(C.x() - radius * std::cos(a), C.y() - radius * std::sin(a), z);
  };

  for (std::size_t fi = 0; fi < frames.size(); ++fi)
  {
    Frame & f = frames[fi];
    auto project_world = [&](const Eigen::Vector3d & pw) {
      return cam.project(f.R_w2c * (pw - f.cam0));
    };
    auto visible = [&](const Eigen::Vector3d & C, double th, int id) {
      const Eigen::Vector3d p = plate_center(C, th, id);
      const double a = th + id * 2.0 * kPi / 4.0;
      const Eigen::Vector3d n(-std::cos(a), -std::sin(a), 0.0);
      return n.dot((f.cam0 - p).normalized()) > 0.0;
    };

    if (!f.dets.empty())
    {
      const double dt = (win.empty() ? 0.0 : std::max(1e-3, f.t - win.back().t));
      if (single_only && f.dets.size() > 1)
      {
        // 只留"看起来最近"的那块（135mm 边长像素最大）
        std::size_t best = 0;
        double bw = -1.0;
        for (std::size_t j = 0; j < f.dets.size(); ++j)
        {
          const double w = 0.5 * ((f.dets[j].p[0] - f.dets[j].p[1]).norm() +
                                  (f.dets[j].p[3] - f.dets[j].p[2]).norm());
          if (w > bw) { bw = w; best = j; }
        }
        f.dets = {f.dets[best]};
      }
      if (!inited)
      {
        const Detection & d = f.dets.front();
        Eigen::Vector2d c = Eigen::Vector2d::Zero();
        for (int k = 0; k < 4; ++k) c += d.p[k];
        c *= 0.25;
        const double wpx = 0.5 * ((d.p[0] - d.p[1]).norm() + (d.p[3] - d.p[2]).norm());
        const double z = cam.fx * kPlateW / std::max(20.0, wpx);
        const Eigen::Vector3d p_cam(
          (c.x() - cam.cx) / cam.fx * z, (c.y() - cam.cy) / cam.fy * z, z);
        const Eigen::Vector3d p_world = f.R_w2c.transpose() * p_cam + f.cam0;
        const Eigen::Vector3d to_cam = f.cam0 - p_world;
        f.th = std::atan2(-to_cam.y(), -to_cam.x());   // 假设这块板正对相机、板号 0
        f.C = plate_center(p_world, f.th, 0);
        f.V.setZero();
        f.w = 0.0;
        inited = true;
      }
      else
      {
        f.C = last_C + last_V * dt;
        f.th = last_th + last_w * dt;
        f.V = last_V;
        f.w = last_w;
      }
      // ---- PnP：给板位姿变量一个"贴着观测"的初值 ---------------------------------
      // 对齐 jlu：PnP 只用来给初值（并提供 roll/pitch），yaw 交给因子去调。
      // 之前用"底盘模型推出来的板心"当初值 —— 观测因子和 tie 因子从第一步就对着干，
      // 双检测帧上直接掉进局部极小（实测底盘被拽到 220px 外）。
      std::vector<Eigen::Vector3d> q(f.dets.size(), Eigen::Vector3d::Zero());
      std::vector<double> qyaw(f.dets.size(), 0.0);
      std::vector<bool> qok(f.dets.size(), false);
      const std::vector<cv::Point3f> op = [&] {
        std::vector<cv::Point3f> v;
        for (int k = 0; k < 4; ++k)
        {
          const auto & o = obj[static_cast<std::size_t>(k)];
          v.emplace_back(
            static_cast<float>(o.x()), static_cast<float>(o.y()), static_cast<float>(o.z()));
        }
        return v;
      }();
      for (std::size_t j = 0; j < f.dets.size(); ++j)
      {
        std::vector<cv::Point2f> ip;
        for (int k = 0; k < 4; ++k)
          ip.emplace_back(
            static_cast<float>(f.dets[j].p[k].x()), static_cast<float>(f.dets[j].p[k].y()));
        cv::Mat rvec, tvec;
        if (!cv::solvePnP(op, ip, camK, camD, rvec, tvec, false, cv::SOLVEPNP_IPPE)) continue;
        cv::Mat Rcv;
        cv::Rodrigues(rvec, Rcv);
        Eigen::Matrix3d Rc;
        cv::cv2eigen(Rcv, Rc);
        Eigen::Vector3d tc;
        cv::cv2eigen(tvec, tc);
        q[j] = f.R_w2c.transpose() * tc + f.cam0;
        const Eigen::Matrix3d Rw = f.R_w2c.transpose() * Rc;
        qyaw[j] = std::atan2(Rw(1, 0), Rw(0, 0));
        qok[j] = true;
      }
      // 两块检测的 PnP 位置几乎重合 → 同一块板被检了两次：只留"看起来更近"的那块
      if (f.dets.size() >= 2 && qok[0] && qok[1] && (q[0] - q[1]).norm() < 0.15)
      {
        const double w0 = 0.5 * ((f.dets[0].p[0] - f.dets[0].p[1]).norm() +
                                 (f.dets[0].p[3] - f.dets[0].p[2]).norm());
        const double w1 = 0.5 * ((f.dets[1].p[0] - f.dets[1].p[1]).norm() +
                                 (f.dets[1].p[3] - f.dets[1].p[2]).norm());
        const std::size_t keep = (w0 >= w1) ? 0 : 1;
        f.dets = {f.dets[keep]};
        q = {q[keep]};
        qyaw = {qyaw[keep]};
        qok = {qok[keep]};
      }

      // ---- 关联 ------------------------------------------------------------------
      // ⚠ 一块模型板在**同一帧只能被认领一次**（否则同帧两块检测都抢 id=0，
      //   两条矛盾约束会把底盘拽飞：实测 V→±3 m/s、ω→-24 rad/s）。
      // 双检测帧用**联合假设**：板号对 (i,j) 一起评，代价里含一项**只由 PnP 决定**的
      // 几何项 —— 两块板的间距必须等于模型里"相邻/对面"的间距（√2·r 或 2r）。
      // 这一项不依赖底盘状态，所以它就是解开"yaw vs yaw+180°"那个分支的信息
      // （见 docs/uv_observation.md 记的镜像位姿问题；也是 rmcs 用"锚点+按 x 序递推"解决的东西）。
      {
        std::vector<Eigen::Vector2d> c(f.dets.size(), Eigen::Vector2d::Zero());
        for (std::size_t j = 0; j < f.dets.size(); ++j)
        {
          for (int k = 0; k < 4; ++k) c[j] += f.dets[j].p[k];
          c[j] *= 0.25;
        }
        auto img_cost = [&](std::size_t j, int id) {
          const double d = (project_world(plate_center(f.C, f.th, id)) - c[j]).norm();
          return (d / 30.0) * (d / 30.0);
        };
        std::vector<int> assign(f.dets.size(), -1);
        const bool pair_ok = (f.dets.size() == 2 && qok[0] && qok[1]);
        double pair_res = std::numeric_limits<double>::infinity();
        if (pair_ok)
        {
          // **解析求 θ**：两块板的 PnP 位置就确定了底盘朝向，不需要预测。
          //   模型：q_k = C − r·u(θ + id_k·90°)，u(a) = (cos a, sin a)
          //   ⇒ q_i − q_j = r·(R(j·90°) − R(i·90°))·u(θ)
          //   ⇒ u(θ) = M⁻¹·(q_i − q_j)/r，θ = atan2(uᵧ, uₓ)，C = q_i + r·u。
          //   12 个有序板号对全部枚举（含镜像/对面），用**图像重投影误差**选最优。
          //   —— 这才是解开"yaw vs yaw+180°/镜像"的信息：间距只能分相邻/对面，
          //     而 PnP 的两点方向能定出 θ，图像再确认是哪一对。
          auto Rot2 = [](double a) {
            Eigen::Matrix2d R;
            R << std::cos(a), -std::sin(a), std::sin(a), std::cos(a);
            return R;
          };
          const double step = 2.0 * kPi / 4.0;
          double best = 1e18;
          int bi = -1, bj = -1;
          Eigen::Vector3d bestC = f.C;
          double best_th = f.th;
          for (int i = 0; i < 4; ++i)
          {
            for (int j = 0; j < 4; ++j)
            {
              if (i == j) continue;
              const Eigen::Matrix2d M = Rot2(j * step) - Rot2(i * step);
              const Eigen::Vector2d dq(q[0].x() - q[1].x(), q[0].y() - q[1].y());
              const Eigen::Vector2d w = M.inverse() * (dq / radius);
              if (!std::isfinite(w.x()) || w.norm() < 1e-6) continue;
              const Eigen::Vector2d u = w.normalized();
              const double th = std::atan2(u.y(), u.x());
              // q_i = C − r·u(θ + i·90°) ⇒ C = q_i + r·u(θ + i·90°)
              const Eigen::Vector2d ui(
                std::cos(th + i * step), std::sin(th + i * step));
              Eigen::Vector3d C = q[0] + radius * Eigen::Vector3d(ui.x(), ui.y(), 0.0);
              // C.z 也要按模型反解（侧板有高度偏移），不能取两块板的均值
              C.z() = q[0].z() - ((i == 1 || i == 3) ? h_offset : 0.0);
              const Eigen::Vector2d P0 = project_world(plate_center(C, th, i));
              const Eigen::Vector2d P1 = project_world(plate_center(C, th, j));
              const double cost = (P0 - c[0]).squaredNorm() + (P1 - c[1]).squaredNorm();
              if (cost < best) { best = cost; bi = i; bj = j; bestC = C; best_th = th; }
            }
          }
          if (bi >= 0)
          {
            pair_res = std::sqrt(std::max(0.0, best) / 2.0);   // 每板残差(px)
            if (pair_res > pair_gate_px)
            {
              // **双板构造残差太大 → 不认这一对**，退回单板路径
              if (std::getenv("FGO_PAIR_DEBUG") != nullptr)
                std::fprintf(
                  stderr, "[pair] t=%.2f 残差 %.1fpx > 门限 %.1fpx → 丢弃这一对，退回单板\n",
                  f.t, pair_res, pair_gate_px);
              bi = bj = -1;
            }
          }
          if (bi >= 0)
          {
            // 用解析解直接初始化这一帧的底盘（比"上一帧外推"好得多）
            f.C = bestC;
            f.th = best_th;
            // 诊断：这一对板在"共用一台 4 板车"假设下的最优图像残差
            if (std::getenv("FGO_PAIR_DEBUG") != nullptr)
            {
              static int pn = 0;
              if (pn++ < 30 || pn % 200 == 0)
                std::fprintf(
                  stderr,
                  "[pair] t=%.2f 最优(id=%d,%d) 每板残差 %.1fpx | PnP 间距 %.3f m | "
                  "自检: |proj(q0)-c0|=%.1fpx  |q0-C-r·u_i|=%.4fm|q1-(C-r·u_j)|=%.4fm\n",
                  f.t, bi, bj, std::sqrt(std::max(0.0, best) / 2.0), (q[0] - q[1]).norm(),
                  (project_world(q[0]) - c[0]).norm(),
                  (q[0] - plate_center(f.C, f.th, bi)).norm(),
                  (q[1] - plate_center(f.C, f.th, bj)).norm());
            }
          }
          assign[0] = bi;
          assign[1] = bj;
        }
        if (assign[0] < 0 || (f.dets.size() >= 2 && assign[1] < 0))
        {
          // ---- 单板路径（含"双板残差太大退回来"的分支）--------------------------
          // 先按"板号 → 该板号下各检测的最小图像代价"给每个可见板号打分，
          // 再套**锚板锁定迟滞**：上一帧锚定的那块只要不明显更差就继续保持
          // （对齐 Ultra_Vision tracker 的 plate_lock_margin=1.35）。
          struct Cand { std::size_t det; int id; double d; };
          double id_cost[4] = {1e18, 1e18, 1e18, 1e18};
          std::size_t id_det[4] = {0, 0, 0, 0};
          for (std::size_t j = 0; j < f.dets.size(); ++j)
            for (int id = 0; id < 4; ++id)
            {
              if (!visible(f.C, f.th, id)) continue;
              const double cst = img_cost(j, id);
              if (cst < id_cost[id]) { id_cost[id] = cst; id_det[id] = j; }
            }
          int best_id = -1;
          for (int id = 0; id < 4; ++id)
            if (id_cost[id] < 1e17 && (best_id < 0 || id_cost[id] < id_cost[best_id])) best_id = id;
          int chosen = best_id;
          if (best_id >= 0 && anchor_id_ >= 0 && anchor_id_ < 4 && id_cost[anchor_id_] < 1e17 &&
              id_cost[anchor_id_] <= plate_lock_margin * id_cost[best_id])
            chosen = anchor_id_;
          if (chosen >= 0)
          {
            assign[id_det[chosen]] = chosen;
            anchor_id_ = chosen;
          }
        }
        for (std::size_t j = 0; j < f.dets.size(); ++j)
        {
          // ⚠ 没被认领的检测**必须丢掉**，不能兜底成 id=0 —— 否则同帧两块又都算到
          //   同一块模型板上（两条矛盾约束），就是前面把底盘拽飞的那个 bug。
          if (assign[j] < 0) continue;
          const int id = assign[j];
          f.plate_id.push_back(id);
          const Eigen::Vector3d p_model = plate_center(f.C, f.th, id);
          // PnP 初值（离底盘模型太远就退回模型值，防止坏 PnP 把局部极小带出来）
          if (qok[j] && (q[j] - p_model).norm() < 1.0)
          {
            f.P.push_back(q[j]);
            f.psi.push_back(qyaw[j]);
          }
          else
          {
            f.P.push_back(p_model);
            f.psi.push_back(f.th + id * 2.0 * kPi / 4.0);
          }
        }
        // 把没被认领的检测从这一帧里删掉（= 不喂给估计器）
        if (f.plate_id.size() != f.dets.size())
        {
          std::vector<Detection> keep;
          std::vector<int> kkeep;
          for (std::size_t j = 0; j < f.dets.size(); ++j)
          {
            if (assign[j] < 0) continue;
            keep.push_back(f.dets[j]);
            kkeep.push_back(assign[j]);
          }
          if (!keep.empty())
          {
            f.dets = keep;
            f.plate_id = kkeep;
            // P/psi 也按同样顺序重建
            std::vector<Eigen::Vector3d> kP;
            std::vector<double> kpsi;
            for (std::size_t j = 0; j < assign.size(); ++j)
            {
              if (assign[j] < 0) continue;
              const int id = assign[j];
              const Eigen::Vector3d p_model = plate_center(f.C, f.th, id);
              if (qok[j] && (q[j] - p_model).norm() < 1.0)
              {
                kP.push_back(q[j]);
                kpsi.push_back(qyaw[j]);
              }
              else
              {
                kP.push_back(p_model);
                kpsi.push_back(f.th + id * 2.0 * kPi / 4.0);
              }
            }
            f.P = kP;
            f.psi = kpsi;
          }
        }
      }

      win.push_back(f);
      if (static_cast<int>(win.size()) > window) win.pop_front();
      // 缺帧太久（>250ms）时窗口里的运动模型没意义 → 直接重开窗口
      if (win.size() >= 2 && (f.t - win[win.size() - 2].t) > 0.25)
      {
        Frame keep = f;
        win.clear();
        win.push_back(keep);
      }

      ceres::Problem prob;
      for (std::size_t k = 0; k < win.size(); ++k)
      {
        Frame & g = win[k];
        prob.AddParameterBlock(g.C.data(), 3);
        prob.AddParameterBlock(g.V.data(), 3);
        prob.AddParameterBlock(&g.th, 1);
        prob.AddParameterBlock(&g.w, 1);
        for (std::size_t j = 0; j < g.P.size(); ++j)
        {
          prob.AddParameterBlock(g.P[j].data(), 3);
          prob.AddParameterBlock(&g.psi[j], 1);
        }
      }
      for (std::size_t k = 0; k < win.size(); ++k)
      {
        Frame & g = win[k];
        for (std::size_t j = 0; j < g.P.size(); ++j)
        {
          const Detection & d = g.dets[j];
          const int id = g.plate_id[j];
          for (int c = 0; c < 4; ++c)
          {
            prob.AddResidualBlock(
              new ceres::AutoDiffCostFunction<ReprojCost, 2, 3, 1>(new ReprojCost(
                cam, g.R_w2c, g.cam0, obj[static_cast<std::size_t>(c)], d.p[c], sigma_px)),
              nullptr, g.P[j].data(), &g.psi[j]);
          }
          const double sig[4] = {0.01, 0.03, 0.01, 0.005};   // 切向/径向/z/相位（对齐 jlu）
          auto * tie = new TieCost(radius, id, 4, sig);
          tie->set_h(h_offset);
          prob.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TieCost, 4, 3, 1, 3, 1>(
              tie),
            nullptr, g.C.data(), &g.th, g.P[j].data(), &g.psi[j]);
        }
      }
      for (std::size_t k = 0; k + 1 < win.size(); ++k)
      {
        Frame & g0 = win[k];
        Frame & g1 = win[k + 1];
        const double ddt = std::max(1e-3, g1.t - g0.t);
        // ⚠ 因子噪声必须**随 dt 缩放**：窗口里只放"有检测的帧"，缺帧时 dt 会到几十 ms
        //   甚至几百 ms。固定 σ 会把"匀速外推"变成硬约束（长间隔时直接把状态拽死）。
        //   这里用物理量表达：位置 σ = ½·a·dt²、速度 σ = a·dt（a=10 m/s²）；
        //   角度 σ = ½·α·dt²、角速度 σ = α·dt（α=50 rad/s²）。
        //   在 dt=14.3ms（70Hz）时正好等于 jlu 用的 0.001 / 0.005。
        constexpr double kAcc = 4.0, kAngAcc = 20.0;
        const double s_pos = 0.5 * kAcc * ddt * ddt;
        const double s_vel = kAcc * ddt;
        const double s_yaw = 0.5 * kAngAcc * ddt * ddt;
        const double s_w = kAngAcc * ddt;
        prob.AddResidualBlock(
          new ceres::AutoDiffCostFunction<MotionCost, 3, 3, 3, 3>(new MotionCost(ddt, s_pos)),
          nullptr, g0.C.data(), g0.V.data(), g1.C.data());
        prob.AddResidualBlock(
          new ceres::AutoDiffCostFunction<VelCost, 3, 3, 3>(new VelCost(s_vel)), nullptr,
          g0.V.data(), g1.V.data());
        prob.AddResidualBlock(
          new ceres::AutoDiffCostFunction<YawMotionCost, 1, 1, 1, 1>(
            new YawMotionCost(ddt, s_yaw)),
          nullptr, &g0.th, &g0.w, &g1.th);
        prob.AddResidualBlock(
          new ceres::AutoDiffCostFunction<WCost, 1, 1, 1>(new WCost(s_w)), nullptr, &g0.w,
          &g1.w);
      }
      {
        Frame & g = win.front();   // 只给最老一帧先验，用来定规范
        prob.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Prior3Cost, 3, 3>(new Prior3Cost(g.C.data(), 0.05)),
          nullptr, g.C.data());
        prob.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Prior3Cost, 3, 3>(new Prior3Cost(g.V.data(), 0.5)),
          nullptr, g.V.data());
        prob.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Prior1Cost, 1, 1>(new Prior1Cost(g.th, 0.05)), nullptr,
          &g.th);
        prob.AddResidualBlock(
          new ceres::AutoDiffCostFunction<Prior1Cost, 1, 1>(new Prior1Cost(g.w, 0.5)), nullptr,
          &g.w);
      }
      ceres::Solver::Options opt;
      opt.linear_solver_type = ceres::DENSE_QR;
      opt.max_num_iterations = 30;
      opt.num_threads = 1;
      opt.logging_type = ceres::SILENT;
      ceres::Solver::Summary summary;
      ceres::Solve(opt, &prob, &summary);

      if (std::getenv("FGO_DEBUG") != nullptr && (fi < 6 || fi % 200 == 0))
      {
        const Frame & g = win.back();
        std::fprintf(
          stderr,
          "[fgo] f=%zu cost %.3e→%.3e iters=%d %s | C=(%.3f,%.3f,%.3f) th=%.3f w=%.3f V=(%.2f,%.2f)\n",
          fi, summary.initial_cost, summary.final_cost, summary.iterations.size(),
          summary.termination_type == ceres::CONVERGENCE ? "CONV" : "NO-CONV", g.C.x(), g.C.y(),
          g.C.z(), g.th, g.w, g.V.x(), g.V.y());
        for (std::size_t j = 0; j < g.P.size(); ++j)
        {
          const Detection & d = g.dets[j];
          const int id = g.plate_id[j];
          double e_var = 0.0, e_chassis = 0.0;
          const Eigen::Vector3d pc = plate_center(g.C, g.th, id);
          const double a = g.th + id * 2.0 * kPi / 4.0;
          for (int c = 0; c < 4; ++c)
          {
            e_var += (project_world(g.P[j] + armor_R(g.psi[j]) * obj[static_cast<std::size_t>(c)]) -
                      d.p[c])
                       .squaredNorm();
            e_chassis +=
              (project_world(pc + armor_R(a) * obj[static_cast<std::size_t>(c)]) - d.p[c])
                .squaredNorm();
          }
          // tie 残差（切向/径向/z/相位）
          const double nx = -std::cos(a), ny = -std::sin(a);
          const double tx = -ny, ty = nx;
          const double dx = g.P[j].x() - g.C.x(), dy = g.P[j].y() - g.C.y();
          std::fprintf(
            stderr,
            "      det%zu id=%d | 观测残差(板变量) %.1fpx  底盘模型 %.1fpx | tie 切向 %.4f 径向 %.4f z %.4f 相位 %.4f\n",
            j, id, std::sqrt(e_var / 8.0), std::sqrt(e_chassis / 8.0), tx * dx + ty * dy,
            nx * dx + ny * dy - radius, g.P[j].z() - g.C.z(), g.psi[j] - a);
        }
      }

      // 把解写回当前帧（窗口里是拷贝）
      f.C = win.back().C;
      f.V = win.back().V;
      f.th = win.back().th;
      f.w = win.back().w;
      f.P = win.back().P;
      f.psi = win.back().psi;
      // 物理区间夹紧（对齐 rmcs 的"状态夹到物理区间"）：解出 |V|>5m/s 或 |ω|>12rad/s
      // 一定是坏的关联/局部极小，直接夹住，避免它经"下一帧的初值"把误差放大成级联失败。
      {
        const double nv = f.V.norm();
        if (nv > 5.0) f.V *= 5.0 / nv;
        if (f.w > 12.0) f.w = 12.0;
        if (f.w < -12.0) f.w = -12.0;
        for (auto & wf : win) { wf.V = f.V; wf.w = f.w; }
      }
      last_C = f.C;
      last_V = f.V;
      last_th = f.th;
      last_w = f.w;
    }

    if (!f.dets.empty() && (lg || dl))
    {
      std::array<Eigen::Vector2d, 4> mc{};
      std::array<std::vector<Eigen::Vector2d>, 4> mq;
      for (int id = 0; id < 4; ++id)
      {
        const Eigen::Vector3d pw = plate_center(f.C, f.th, id);
        const double a = f.th + id * 2.0 * kPi / 4.0;
        for (int c = 0; c < 4; ++c)
          mq[static_cast<std::size_t>(id)].push_back(
            project_world(pw + armor_R(a) * obj[static_cast<std::size_t>(c)]));
        Eigen::Vector2d s = Eigen::Vector2d::Zero();
        for (const auto & q : mq[static_cast<std::size_t>(id)]) s += q;
        mc[static_cast<std::size_t>(id)] = s * 0.25;
      }
      double du = 0, dv = 0;
      for (const auto & d : f.dets)
      {
        Eigen::Vector2d c = Eigen::Vector2d::Zero();
        for (int k = 0; k < 4; ++k) c += d.p[k];
        du += c.x() * 0.25;
        dv += c.y() * 0.25;
      }
      du /= static_cast<double>(f.dets.size());
      dv /= static_cast<double>(f.dets.size());
      if (lg)
      {
        lg << std::setprecision(9) << f.t << ",1," << f.dets.size() << ',' << f.th << ',' << f.w
           << ',' << f.C.x() << ',' << f.C.y() << ',' << f.dets.size() << ',' << du << ',' << dv;
        for (int id = 0; id < 4; ++id)
          lg << ',' << mc[static_cast<std::size_t>(id)].x() << ','
             << mc[static_cast<std::size_t>(id)].y();
        lg << ",nan," << radius << ",0,0\n";
      }
      if (dl)
      {
        for (std::size_t j = 0; j < f.dets.size(); ++j)
        {
          const Detection & d = f.dets[j];
          Eigen::Vector2d c = Eigen::Vector2d::Zero();
          for (int k = 0; k < 4; ++k) c += d.p[k];
          c *= 0.25;
          int bi = 0;
          double bd = 1e18;
          for (int id = 0; id < 4; ++id)
          {
            const double e = (mc[static_cast<std::size_t>(id)] - c).squaredNorm();
            if (e < bd) { bd = e; bi = id; }
          }
          double s = 0.0;
          for (int k = 0; k < 4; ++k)
            s += (mq[static_cast<std::size_t>(bi)][static_cast<std::size_t>(k)] - d.p[k])
                   .squaredNorm();
          const auto & q = mq[static_cast<std::size_t>(bi)];
          const double w135 = 0.5 * ((q[0] - q[1]).norm() + (q[2] - q[3]).norm());
          const double h55 = 0.5 * ((q[1] - q[2]).norm() + (q[3] - q[0]).norm());
          dl << std::setprecision(9) << f.t << ',' << fi << ',' << f.dets.size() << ',' << j
             << ',' << c.x() << ',' << c.y() << ',' << bi << ',' << std::sqrt(bd) << ','
             << std::sqrt(s / 8.0) << ',' << f.w << ',' << f.C.x() << ',' << f.C.y() << ','
             << f.th << ',' << f.P[j].x() << ',' << f.P[j].y() << ',' << f.P[j].z() << ','
             << f.psi[j] << ",nan,nan,nan," << w135 << ',' << h55 << '\n';
        }
      }
    }
  }
  std::printf("[fgo] 完成\n");
  return 0;
}
