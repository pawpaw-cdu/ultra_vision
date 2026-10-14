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
      res[2] = (P[2] - C[2]) / T(sig_[2]);
      res[3] = (psi[0] - a) / T(sig_[3]);
      return true;
    }
    double radius_, sig_[4];
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
  bool single_only = false;   // 只用"最近的那一块"检测（诊断双检测的影响）
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    if (a == "--detections" && i + 1 < argc) det_path = argv[++i];
    else if (a == "--camera" && i + 1 < argc) cam_path = argv[++i];
    else if (a == "--extrinsics" && i + 1 < argc) ext_path = argv[++i];
    else if (a == "--window" && i + 1 < argc) window = std::atoi(argv[++i]);
    else if (a == "--radius" && i + 1 < argc) radius = std::atof(argv[++i]);
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
  {
    const YAML::Node root = YAML::LoadFile(cam_path);
    const YAML::Node sec = root["hikcamera"] ? root["hikcamera"] : root["galaxy"];
    const auto K = readVec(sec["camera_matrix"]);
    const auto D = readVec(sec["dist_coeffs"] ? sec["dist_coeffs"] : sec["distort_coeffs"]);
    cam.fx = K[0]; cam.fy = K[4]; cam.cx = K[2]; cam.cy = K[5];
    if (D.size() >= 5) { cam.k1 = D[0]; cam.k2 = D[1]; cam.p1 = D[2]; cam.p2 = D[3]; cam.k3 = D[4]; }
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

  auto plate_center = [&](const Eigen::Vector3d & C, double th, int id) {
    const double a = th + id * 2.0 * kPi / 4.0;
    return Eigen::Vector3d(C.x() - radius * std::cos(a), C.y() - radius * std::sin(a), C.z());
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
      // 关联：图像最近邻 + 可见性
      // ⚠ 一块模型板在**同一帧只能被认领一次**（贪心按图像距离）。
      //   否则同帧两块检测可能都抢到 id=0，两条矛盾的约束会把底盘直接拽飞
      //   （实测：f≈400 起 V 冲到 ±3 m/s、ω 到 -24 rad/s，就是从这里开始的）。
      {
        struct Cand { std::size_t det; int id; double d; };
        std::vector<Cand> cands;
        for (std::size_t j = 0; j < f.dets.size(); ++j)
        {
          Eigen::Vector2d c = Eigen::Vector2d::Zero();
          for (int k = 0; k < 4; ++k) c += f.dets[j].p[k];
          c *= 0.25;
          for (int id = 0; id < 4; ++id)
          {
            if (!visible(f.C, f.th, id)) continue;
            cands.push_back({j, id, (project_world(plate_center(f.C, f.th, id)) - c).squaredNorm()});
          }
        }
        std::sort(cands.begin(), cands.end(), [](const Cand & a, const Cand & b) { return a.d < b.d; });
        std::vector<int> assign(f.dets.size(), -1);
        bool used[4] = {false, false, false, false};
        for (const Cand & c : cands)
        {
          if (assign[c.det] >= 0 || used[c.id]) continue;
          assign[c.det] = c.id;
          used[c.id] = true;
        }
        for (std::size_t j = 0; j < f.dets.size(); ++j)
        {
          const int id = assign[j] >= 0 ? assign[j] : 0;
          f.plate_id.push_back(id);
          f.P.push_back(plate_center(f.C, f.th, id));
          f.psi.push_back(f.th + id * 2.0 * kPi / 4.0);
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
          prob.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TieCost, 4, 3, 1, 3, 1>(
              new TieCost(radius, id, 4, sig)),
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
