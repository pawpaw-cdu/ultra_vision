// 用 **sp_vision_25 的实现**（src/spv/：solver + tracker + target/EKF）回放实车检测流，
// 目的是给"底盘模型构造 + 重投影"换一个用户已在真机上验证过的底座，然后只在这个底座上
// 做局部拟合器的 A/B —— 不再拿我们自己那套打过很多补丁的 tracker 做实验。
//
// 输入：real_runner 落的 ULTRA_VISION_DETECTIONS_CSV（一行一个检测，含云台角、板面数字、
//       四角；没有检测的帧也有一行 n_det=0）。
// 输出：--out 逐帧状态（sp_vision 的 11 维状态 + 重投影误差）。
//
// 坐标系对齐（关键）：
//   sp_vision 的链路是  p_world = R_gimbal2world · (R_camera2gimbal · p_cam + t_camera2gimbal)
//   我们的是           p_world = (base_to_world · cameraToBaseRotation(yaw,pitch) · cameraToGimbal) · p_cam
//   所以令 R_camera2gimbal = cameraToGimbal（手眼）、t = 手眼平移、
//   而 R_gimbal2world = base_to_world · cameraToBaseRotation(yaw,pitch)（每帧更新）。
//   这样两条链路逐位等价，sp_vision 的 Solver/Tracker 拿到的就是真机同一个观测。
//
// 物点尺寸：默认用环境变量覆盖成 Ultra_Vision 自己标定的等效尺寸（131.5/60mm），
//   否则 sp_vision 原值 135/56mm 会带来 4~7% 的距离系统偏差，A/B 不公平。
//   要看 sp_vision 原值：不设 SPV_* 环境变量即可。

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "Kalman/tracker.hpp"          // 复用它的 CameraPose 定义
#include "Kalman/tracker_utils.hpp"    // cameraToGimbal / cameraToBaseRotation
#include "config_loader.hpp"
#include "spv/solver.hpp"
#include "spv/tracker.hpp"
#include "spv/tools/extended_kalman_filter.hpp"
#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
#include "estimation/local_trajectory_fitter.hpp"
#endif

namespace
{
    std::vector<std::string> splitCsv(const std::string& line)
    {
        std::vector<std::string> out;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) out.push_back(cell);
        return out;
    }
    std::string trim(const std::string& s)
    {
        const auto b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return {};
        const auto e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }
    bool cell(const std::vector<std::string>& r, int i, double& v)
    {
        if (i < 0 || i >= static_cast<int>(r.size())) return false;
        const std::string s = trim(r[static_cast<std::size_t>(i)]);
        if (s.empty() || s == "nan" || s == "None") return false;
        try { v = std::stod(s); } catch (...) { return false; }
        return std::isfinite(v);
    }

    /// 把我们的 CameraPose 变成 sp_vision 需要的 R_gimbal2world。
    Eigen::Matrix3d gimbalToWorld(const auto_aim::CameraPose& pose)
    {
        Eigen::Matrix3d base_to_world;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) base_to_world(r, c) = pose.base_to_world[r][c];
        const auto cb = auto_aim::tracker_detail::cameraToBaseRotation(pose.yaw, pose.pitch);
        Eigen::Matrix3d cb_eigen;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) cb_eigen(r, c) = cb(r, c);
        return base_to_world * cb_eigen;
    }

    struct Det
    {
        std::array<cv::Point2f, 4> corners{};   // TL,TR,BR,BL（sp_vision 的顺序）
        int number = -1;
        bool valid = false;
    };
    struct Frame
    {
        double yaw = 0.0;
        double pitch = 0.0;
        std::vector<Det> dets;
    };
}

int main(int argc, char* argv[])
{
    std::string det_path;
    std::string config_dir = "configs";
    std::string out_path;
    int max_rows = 0;
    bool local_fit = false;
    bool use_yaw = false;
    int fit_period = 8;
    double omega_sign = -1.0;   // 拟合器 ω 与 EKF ω 的符号（真机标定项：-1）
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--detections" && i + 1 < argc) det_path = argv[++i];
        else if (a == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--max-rows" && i + 1 < argc) max_rows = std::atoi(argv[++i]);
        else if (a == "--local-fit") local_fit = true;
        else if (a == "--use-yaw") use_yaw = true;
        else if (a == "--fit-period" && i + 1 < argc) fit_period = std::max(1, std::atoi(argv[++i]));
        else if (a == "--omega-sign" && i + 1 < argc) omega_sign = std::atof(argv[++i]);
        else { std::cerr << "未知参数 " << a << std::endl; return 2; }
    }
    if (det_path.empty())
    {
        std::cerr << "用法：spv_replay --detections <detections.csv> [--config-dir configs]\n"
                     "                 [--out dump.csv] [--max-rows N]\n";
        return 2;
    }

    // ---- 相机内参 + 手眼外参（喂给 sp_vision 的 Solver）----------------------
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const std::string cam_name = camera_file["camera"]["name"].as<std::string>("galaxy");
    const YAML::Node intr = (cam_name == "hikcamera") ? camera_file["hikcamera"]
                                                      : camera_file["galaxy"];
    // 默认用我们自己的内参；若 spv_extrinsics.yaml 里也给了（现场那份有），则改用它 ——
    // 这样 PnP（spv 侧）与拟合器去畸变走的是同一组内参，自洽。
    cv::Mat K = auto_aim::readMatFromYaml(intr["camera_matrix"]);
    cv::Mat D = auto_aim::readMatFromYaml(intr["dist_coeffs"]);
    Eigen::Matrix3d R_g2ib = Eigen::Matrix3d::Identity();

    auto_aim::CameraPose hand_eye;   // 只为取手眼外参；姿态每帧覆盖
    // 相机内外参：**用 sp_vision 自己那份**（--spv-yaml，默认 configs/spv_extrinsics.yaml，
    // 它就是 NUC 上实跑的 sp_vision_25/configs/standard3.yaml 里的值）。
    // 为什么不用 configs/camera.yaml 那组：实测把那份 R_camera2gimbal 喂给 sp_vision，
    // 重投影指标从 3.4px 掉到 149px（同一份数据）—— 约定与 sp_vision 期望的不一致。
    // NUC 上这份是现场验证过能让 sp_vision 跑出好效果的那一组。
    const std::string spv_yaml = [&] {
        const char* e = std::getenv("SPV_YAML");
        return e ? std::string(e) : (config_dir + "/spv_extrinsics.yaml");
    }();
    {
        const YAML::Node sp = YAML::LoadFile(spv_yaml);
        // ⚠ 默认**不**用 spv 的 R_camera2gimbal：它是给 sp_vision 自己那套云台系
        // （x 前 / y 左 / z 上）标的，而我们的 cameraToBaseRotation 输出的是我们的
        // 体坐标系（x 右 / y 下 / z 前）。两者相差一个固定的 90° 置换 Q；直接拼
        // 会把世界系的"竖直"拧歪，实测重投影指标 3.4px → 27px（同一份数据）。
        // 要用就设 SPV_USE_SPV_EXTRINSICS=1（那时得同时把云台角→世界系那一步也换成
        // sp_vision 的口径，最忠实的做法是直接录 C 板 IMU 四元数喂 set_R_gimbal2world）。
        const bool use_spv_extrinsics = (std::getenv("SPV_USE_SPV_EXTRINSICS") != nullptr);
        if (use_spv_extrinsics && sp["R_camera2gimbal"])
        {
            const auto r = sp["R_camera2gimbal"].as<std::vector<double>>();
            for (int i = 0; i < 3 && r.size() == 9; ++i)
                for (int j = 0; j < 3; ++j)
                    hand_eye.camera_to_gimbal[static_cast<std::size_t>(i)]
                                             [static_cast<std::size_t>(j)] =
                        r[static_cast<std::size_t>(i * 3 + j)];
        }
        if (use_spv_extrinsics && sp["t_camera2gimbal"])
        {
            const auto t = sp["t_camera2gimbal"].as<std::vector<double>>();
            for (int i = 0; i < 3 && t.size() == 3; ++i)
                hand_eye.camera_to_gimbal_translation[static_cast<std::size_t>(i)] =
                    t[static_cast<std::size_t>(i)];
        }
        // 内参/畸变：**默认用我们自己的**（我们的物点尺寸 131.5/60 是配着这组标定的，
        // 混用会让 PnP 距离带上 2~3% 的系统偏差 → 重投影直接差几十 px）。
        // 要用 spv 那份现场内参就设 SPV_USE_SPV_INTRINSICS=1（那时物点尺寸也该跟着设成
        // SPV_SMALL_ARMOR_WIDTH=0.135 SPV_LIGHTBAR_LEN=0.056，保持同一套）。
        const bool use_spv_intrinsics = (std::getenv("SPV_USE_SPV_INTRINSICS") != nullptr);
        if (use_spv_intrinsics && sp["camera_matrix"] && sp["camera_matrix"].size() == 9)
        {
            const auto m = sp["camera_matrix"].as<std::vector<double>>();
            K = (cv::Mat_<double>(3, 3) << m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
        }
        if (use_spv_intrinsics && sp["distort_coeffs"])
        {
            const auto d = sp["distort_coeffs"].as<std::vector<double>>();
            D = cv::Mat(static_cast<int>(d.size()), 1, CV_64F);
            for (std::size_t i = 0; i < d.size(); ++i)
                D.at<double>(static_cast<int>(i), 0) = d[i];
        }
        // R_gimbal2imubody（现场那份是单位阵；非单位阵时下面按 spv 的公式折算）。
        if (sp["R_gimbal2imubody"])
        {
            const auto r = sp["R_gimbal2imubody"].as<std::vector<double>>();
            for (int i = 0; i < 3 && r.size() == 9; ++i)
                for (int j = 0; j < 3; ++j)
                    R_g2ib(i, j) = r[static_cast<std::size_t>(i * 3 + j)];
        }
        std::cout << "[spv] 外参来自 " << spv_yaml << std::endl;
        std::cout << "[spv] fx=" << K.at<double>(0, 0) << " cx=" << K.at<double>(0, 2)
                  << " cy=" << K.at<double>(1, 2) << std::endl;
    }
    const double yaw_sign = [&] {
        const YAML::Node s = YAML::LoadFile(config_dir + "/serial.yaml");
        const YAML::Node n = s["serial"] ? s["serial"] : s;
        return n["yaw_sign"].as<double>(1.0);
    }();
    const double pitch_sign = [&] {
        const YAML::Node s = YAML::LoadFile(config_dir + "/serial.yaml");
        const YAML::Node n = s["serial"] ? s["serial"] : s;
        return n["pitch_sign"].as<double>(1.0);
    }();

    // 生成 sp_vision 需要的配置文件（它只吃路径）。
    const std::string cfg_path = "/tmp/spv_replay_config.yaml";
    {
        std::ofstream f(cfg_path, std::ios::out | std::ios::trunc);
        // 注意：sp_vision 的 Tracker::track() **忽略 use_enemy_color 参数、无条件按颜色过滤**
        // （原版就是这么写的）。所以这里必须和下面建 Armor 时给的 color_id 一致：
        // color_id=1 → red，于是 enemy_color 也必须是 red，否则所有检测会被 remove_if 掉。
        f << "enemy_color: \"red\"\n";
        f << "min_detect_count: 3\nmax_temp_lost_count: 25\noutpost_max_temp_lost_count: 75\n";
        f << "R_gimbal2imubody: [1,0,0, 0,1,0, 0,0,1]\n";
        f << std::setprecision(12);
        f << "camera_matrix: [" << K.at<double>(0, 0) << ",0," << K.at<double>(0, 2) << ",0,"
          << K.at<double>(1, 1) << "," << K.at<double>(1, 2) << ",0,0,1]\n";
        f << "distort_coeffs: [";
        for (int i = 0; i < 5; ++i) f << (i ? "," : "") << (i < D.total() ? D.at<double>(i) : 0.0);
        f << "]\n";
        f << "R_camera2gimbal: [";
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                f << ((r || c) ? "," : "") << hand_eye.camera_to_gimbal[r][c];
        f << "]\n";
        f << "t_camera2gimbal: [" << hand_eye.camera_to_gimbal_translation[0] << ","
          << hand_eye.camera_to_gimbal_translation[1] << ","
          << hand_eye.camera_to_gimbal_translation[2] << "]\n";
    }
    std::cout << "[spv] config=" << cfg_path << "  detections=" << det_path << std::endl;
    std::cout << "[spv] 物点尺寸: 灯条="
              << (std::getenv("SPV_LIGHTBAR_LEN") ? std::atof(std::getenv("SPV_LIGHTBAR_LEN"))
                                                  : 0.056)
              << " 小板宽="
              << (std::getenv("SPV_SMALL_ARMOR_WIDTH")
                      ? std::atof(std::getenv("SPV_SMALL_ARMOR_WIDTH"))
                      : 0.135)
              << " (不设 SPV_* 就是 sp_vision 原值)" << std::endl;

    spv::Solver solver(cfg_path);
    spv::Tracker tracker(cfg_path, solver);

#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
    // ---- 局部拟合器（要验证的主角）--------------------------------------------
    // 与真机同一套口径：相机系里拟合出 (θ, ω)，再换算到 home 系注入 spv 的 EKF。
    // 注入点：spv 的状态就是 [x,vx,y,vy,z,vz,a,w,r,l,h] —— a/w 直接就是我们要的
    // 底盘 yaw 与角速度，所以一路 1×11 的 H 就够（不像我们原来那套还要处理 sin/cos）。
    std::unique_ptr<auto_aim::LocalTrajectoryFitter> fitter;
    auto_aim::LocalTrajectoryFitConfig fcfg;
    if (local_fit)
    {
        const auto env_d = [](const char* n, double d) {
            const char* v = std::getenv(n);
            return v != nullptr ? std::atof(v) : d;
        };
        fcfg.K = {K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2),
                  K.at<double>(1, 2)};
        fcfg.lightbar_len = env_d("SPV_LIGHTBAR_LEN", 0.056);
        fcfg.plate_width = env_d("SPV_SMALL_ARMOR_WIDTH", 0.135);
        fcfg.max_rmse_deg = env_d("ULTRA_VISION_LOCAL_FIT_MAX_RMSE_DEG", 0.4);
        fcfg.gimbal_pitch = 0.0;
        fcfg.center_init = {{0.0, 0.0, 3.0}};
        fitter = std::make_unique<auto_aim::LocalTrajectoryFitter>(fcfg);
        const double alpha0 = env_d("ULTRA_VISION_LOCAL_FIT_ALPHA_DEG", 15.0) * M_PI / 180.0;
        fitter->seedSlow(alpha0, fcfg.R_prior);
        std::cout << "[spv] local-fit ON (alpha=" << env_d("ULTRA_VISION_LOCAL_FIT_ALPHA_DEG", 15.0)
                  << "deg, use_yaw=" << (use_yaw ? 1 : 0) << ")" << std::endl;
    }
#else
    if (local_fit)
        std::cerr << "[spv] 本次构建没有局部拟合器（未找到 Ceres）" << std::endl;
#endif

    // ---- 读检测流 -------------------------------------------------------------
    std::ifstream in(det_path);
    if (!in) { std::cerr << "打不开 " << det_path << std::endl; return 1; }
    std::map<long long, Frame> frames;
    {
        std::string line;
        std::getline(in, line);
        const auto head = splitCsv(line);
        std::map<std::string, int> col;
        for (std::size_t i = 0; i < head.size(); ++i) col[trim(head[i])] = static_cast<int>(i);
        const auto of = [&](const std::string& n) {
            const auto it = col.find(n);
            return it == col.end() ? -1 : it->second;
        };
        const int i_t = of("time_us"), i_y = of("cmd_yaw"), i_p = of("cmd_pitch");
        const int i_num = of("armor_number");
        std::array<int, 4> cu{}, cvv{};
        for (int k = 0; k < 4; ++k)
        {
            cu[static_cast<std::size_t>(k)] = of("u" + std::to_string(k));
            cvv[static_cast<std::size_t>(k)] = of("v" + std::to_string(k));
        }
        while (std::getline(in, line))
        {
            if (line.empty()) continue;
            const auto r = splitCsv(line);
            double tu = 0.0;
            if (!cell(r, i_t, tu)) continue;
            Frame& fr = frames[static_cast<long long>(std::llround(tu))];
            double y = 0.0, p = 0.0, num = -1.0;
            if (cell(r, i_y, y)) fr.yaw = y;
            if (cell(r, i_p, p)) fr.pitch = p;
            cell(r, i_num, num);
            Det d;
            d.number = static_cast<int>(num);
            d.valid = true;
            for (int k = 0; k < 4; ++k)
            {
                double u = 0.0, v = 0.0;
                if (!cell(r, cu[static_cast<std::size_t>(k)], u) ||
                    !cell(r, cvv[static_cast<std::size_t>(k)], v))
                {
                    d.valid = false;
                    break;
                }
                // 我们的顺序 = 左下/左上/右上/右下 → sp_vision 要 TL/TR/BR/BL
                d.corners[static_cast<std::size_t>(k)] = cv::Point2f(static_cast<float>(u),
                                                                     static_cast<float>(v));
            }
            if (d.valid)
            {
                // 重排：我们的 [0]=左下 [1]=左上 [2]=右上 [3]=右下
                const std::array<cv::Point2f, 4> src = d.corners;
                d.corners = {src[1], src[2], src[3], src[0]};   // TL,TR,BR,BL
                fr.dets.push_back(d);
            }
        }
    }

    {
        std::size_t nd = 0;
        for (const auto& kv : frames) nd += kv.second.dets.size();
        std::cout << "[spv] 帧=" << frames.size() << " 检测=" << nd;
        if (!frames.empty()) std::cout << " 首帧检测=" << frames.begin()->second.dets.size();
        std::cout << std::endl;
    }

    std::ofstream out;
    if (!out_path.empty())
    {
        out.open(out_path, std::ios::out | std::ios::trunc);
        // 尾部追加 EKF 状态协方差 P 的对角线（σ²）与 NIS：用来判断"滤波器自己
        // 报的不确定度是否真实"。spv 的状态是 [x,vx,y,vy,z,vz,a,w,r,l,h]。
        out << "t,state,n_det,sx,sy,sz,yaw,omega,r,l,h,detfit_med,detfit_max";
        for (int i = 0; i < 11; ++i) out << ",P" << i;
        out << ",nis,nis_pre\n";
    }

    const auto base_time = std::chrono::steady_clock::now();
    long long first_us = 0;
    bool have_first = false;
    int rows = 0;
    int tracked = 0;
    int fit_tried = 0;
    int fit_injected = 0;
    std::vector<auto_aim::LightBarObservation> fit_window;
    // 单板弧的状态（图像板心 + 常速预测），见上面的关联逻辑。
    bool have_arc = false;
    double arc_u = 0.0, arc_v = 0.0, arc_vu = 0.0, arc_vv = 0.0, arc_t = 0.0;
    std::vector<double> detfit;
    std::vector<double> detfit_worst;

    for (const auto& [us, fr] : frames)
    {
        if (max_rows > 0 && rows >= max_rows) break;
        if (!have_first) { first_us = us; have_first = true; }
        const double t = static_cast<double>(us - first_us) * 1e-6;
        auto_aim::CameraPose pose = hand_eye;   // 继承手眼外参
        pose.yaw = yaw_sign * fr.yaw;
        pose.pitch = pitch_sign * fr.pitch;
        // spv 的链路是 R_g2w·(R_c2g·p + t)；要让它的**总变换**等于我们自己的
        // world←camera（W = base_to_world·cameraToBaseRotation），必须取
        // R_g2w = W · R_c2g^T（否则外参会重复/抵消错，实测重投影 3.4px→149px）。
        const Eigen::Matrix3d W_cam = gimbalToWorld(pose);
        const Eigen::Matrix3d R_c2g = [&] {
            Eigen::Matrix3d m;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    m(i, j) = pose.camera_to_gimbal[static_cast<std::size_t>(i)]
                                                  [static_cast<std::size_t>(j)];
            return m;
        }();
        const Eigen::Matrix3d R_g2w_desired = W_cam * R_c2g.transpose();
        // spv 内部做 R_g2ib^T·q·R_g2ib，所以要先把 q 折算过去（R_g2ib=I 时就是它本身）。
        solver.set_R_gimbal2world(
            Eigen::Quaterniond(R_g2ib * R_g2w_desired * R_g2ib.transpose()));

        std::list<spv::Armor> armors;
        for (const Det& d : fr.dets)
        {
            // color_id=1(red) 随便给；用 use_enemy_color=false 关掉颜色过滤。
            armors.emplace_back(1, d.number, 1.0f, cv::Rect(),
                                std::vector<cv::Point2f>{d.corners[0], d.corners[1], d.corners[2],
                                                         d.corners[3]});
        }
        const auto tp = base_time + std::chrono::microseconds(us - first_us);
        auto targets = tracker.track(armors, tp, /*use_enemy_color=*/false);

#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
        if (fitter)
        {
            spv::Target& tgt = tracker.target_mutable();
            // 拟合器的初值：**用 EKF 当前中心热启动**（世界系 → 相机系）。
            // 真机原来固定 (0,0,3) 而靶在 1.2~1.5m，实测拟合会掉进坏局部解
            // （rmse 96px、166 次全部无效）。
            Eigen::Matrix3d R_g2w_cur = Eigen::Matrix3d::Identity();
            if (!targets.empty())
            {
                const Eigen::VectorXd xs = tgt.ekf_x();
                Eigen::Vector3d cw(xs[0], xs[2], xs[4]);
                const Eigen::Matrix3d W = gimbalToWorld(pose);
                const Eigen::Vector3d cc = W.transpose() * cw;
                auto c0 = fitter->config();
                c0.center_init = {{cc[0], cc[1], cc[2]}};
                fitter->setConfig(c0);
            }
            // 0) 先按当前姿态更新拟合器的"车辆系→相机系"完整旋转（含云台 pitch/手眼）。
            {
                auto c = fitter->config();
                c.use_rcv = true;
                const auto cb = auto_aim::tracker_detail::cameraToBaseRotation(pose.yaw, pose.pitch);
                const auto g = auto_aim::tracker_detail::cameraToGimbal(pose);
                const double B[3][3] = {{0.0, 1.0, 0.0}, {0.0, 0.0, -1.0}, {1.0, 0.0, 0.0}};
                double M[3][3];
                for (int i = 0; i < 3; ++i)
                    for (int l = 0; l < 3; ++l)
                    {
                        double v = 0.0;
                        for (int k = 0; k < 3; ++k) v += g(k, i) * cb(l, k);
                        M[i][l] = v;
                    }
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                    {
                        double v = 0.0;
                        for (int l = 0; l < 3; ++l) v += M[i][l] * B[l][j];
                        c.rcv[i * 3 + j] = v;
                    }
                fitter->setConfig(c);
            }
            // 1) 把本帧每块检测的两根灯条端点推进缓冲（在关联之前，与真机同序）。
            //    缓冲自己维护（sp_vision 侧没有这个概念）：只保留最近 window 帧。
            // 只喂**一块板**的弧：按图像板心"最近邻 + 常速预测"关联（与真机
            // Tracker::pushArmorLightBars 的 A 段同构）。把同帧多块板混进一条弧，
            // 拟合器会拿两块板当一块去拟合 → rmse 直接爆掉（实测 fit=0/172）。
            int pick = -1;
            double best = 1e18;
            for (std::size_t i = 0; i < fr.dets.size(); ++i)
            {
                const auto& d = fr.dets[static_cast<std::size_t>(i)];
                const cv::Point2f c = 0.25f * (d.corners[0] + d.corners[1] + d.corners[2] +
                                               d.corners[3]);
                const double dt = t - arc_t;
                const double pu = arc_u + (have_arc && dt > 0.0 ? arc_vu * dt : 0.0);
                const double pv = arc_v + (have_arc && dt > 0.0 ? arc_vv * dt : 0.0);
                const double dist = have_arc ? std::hypot(c.x - pu, c.y - pv) : 0.0;
                if (have_arc && dist > 45.0) continue;
                if (dist < best) { best = dist; pick = static_cast<int>(i); }
            }
            if (pick >= 0)
            {
                const Det& d = fr.dets[static_cast<std::size_t>(pick)];
                const cv::Point2f c = 0.25f * (d.corners[0] + d.corners[1] + d.corners[2] +
                                               d.corners[3]);
                if (have_arc && t > arc_t)
                {
                    const double dt = t - arc_t;
                    arc_vu = 0.5 * (c.x - arc_u) / dt + 0.5 * arc_vu;
                    arc_vv = 0.5 * (c.y - arc_v) / dt + 0.5 * arc_vv;
                }
                arc_u = c.x; arc_v = c.y; arc_t = t;
                have_arc = true;
                // 我们的角点顺序：0=左下 1=左上 2=右上 3=右下
                const std::array<std::array<cv::Point2f, 2>, 2> bars{{
                    {{d.corners[1], d.corners[0]}},   // 左灯条：上、下
                    {{d.corners[2], d.corners[3]}},   // 右灯条：上、下
                }};
                const std::array<double, 2> off{{+0.5 * fcfg.plate_width,
                                                 -0.5 * fcfg.plate_width}};
            for (int b = 0; b < 2; ++b)
            {
                auto_aim::LightBarObservation o;
                o.timestamp = t;
                // **必须先去畸变**：拟合器是纯针孔模型，而真机 k2=0.336/k3=−1.24 很显著；
                // 不去畸变实测 rmse 直接 96px（有效解 0/166）。（spv 的 Solver 自己吃
                // distort_coeffs，所以喂给它的仍是原始角点。）
                const std::vector<cv::Point2f> in{bars[static_cast<std::size_t>(b)][0],
                                                  bars[static_cast<std::size_t>(b)][1]};
                std::vector<cv::Point2f> und;
                if (!D.empty()) cv::undistortPoints(in, und, K, D, cv::noArray(), K);
                else und = in;
                o.u_top = und[0].x;
                o.v_top = und[0].y;
                o.u_bot = und[1].x;
                o.v_bot = und[1].y;
                    o.armor_id = 0;   // 单板弧：板号只影响相位代表元，注入时按模 90° 消掉
                    o.bar_offset = off[static_cast<std::size_t>(b)];
                    o.confidence = 1.0;
                    fit_window.push_back(o);
                }
            }
            // 弧断太久（>0.3s）→ 清窗重来（θ(t) 线性模型跨太大相位会出镜像解）。
            if (have_arc && t - arc_t > 0.30) { fit_window.clear(); have_arc = false; }
            const std::size_t cap = 2 * 24;   // window_frames=24，每帧 2 根
            if (fit_window.size() > cap)
                fit_window.erase(fit_window.begin(),
                                 fit_window.begin() + static_cast<long>(fit_window.size() - cap));
            // 2) 低频拟合 + 注入
            if (rows % fit_period == 0 && fit_window.size() >= 8)
            {
                auto_aim::LocalFitResult r;
                if (fitter->fitWithSlow(fit_window, r) && r.valid && !targets.empty())
                {
                    auto& ekf = tgt.ekf_mutable();
                    // ω：他们的状态第 7 位直接就是角速度。
                    Eigen::VectorXd z(1);
                    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 11);
                    Eigen::MatrixXd Rn = Eigen::MatrixXd::Identity(1, 1);
                    z[0] = r.omega * omega_sign;
                    H(0, 7) = 1.0;
                    Rn(0, 0) = 0.05;
                    ekf.update(z, H, Rn);
                    if (use_yaw)
                    {
                        // yaw：他们的 a 与我们拟合的 θ 相差 "π + k·90°"（两套板号约定的
                        // 反射 + 180°），用模 90° 吸附一次把不确定的常数全部消掉。
                        const double a_target = r.yaw + M_PI;
                        z[0] = ekf.x[6] + std::remainder(a_target - ekf.x[6], 0.5 * M_PI);
                        H.setZero();
                        H(0, 6) = 1.0;
                        Rn(0, 0) = 0.02;
                        const auto sub = [](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                            Eigen::VectorXd c = a - b;
                            c[0] = std::remainder(c[0], 2 * M_PI);
                            return c;
                        };
                        ekf.update(z, H, Rn, sub);
                    }
                    ++fit_injected;
                }
                else if (std::getenv("SPV_REPLAY_DBG") && fit_tried < 6)
                    std::cerr << "[fit-dbg] rows=" << rows << " nbars=" << fit_window.size()
                              << " valid=" << (r.valid ? 1 : 0) << " rmse=" << r.reprojection_rmse_px
                              << " yaw=" << r.yaw << " w=" << r.omega << " R=" << r.R
                              << " targets=" << targets.size() << std::endl;
                ++fit_tried;
            }
        }
#endif

        if (std::getenv("SPV_REPLAY_DBG") && rows < 8 && !armors.empty())
        {
            const auto& a = armors.front();
            std::cerr << "[dbg] frame " << rows << " state=" << tracker.state() << " n=" << armors.size()
                      << " name=" << static_cast<int>(a.name) << " type=" << static_cast<int>(a.type)
                      << " xyz_w=(" << a.xyz_in_world.transpose() << ")"
                      << " ypd_w=(" << a.ypd_in_world.transpose() << ")"
                      << " targets=" << targets.size() << std::endl;
        }

        double sx = 0, sy = 0, sz = 0, yaw = 0, omega = 0, rr = 0, ll = 0, hh = 0;
        double med = -1.0, mx = -1.0;
        std::array<double, 11> pdiag{};
        double nis = -1.0;
        double nis_pre = -1.0;
        if (!targets.empty())
        {
            const spv::Target& tgt = targets.front();
            const Eigen::VectorXd x = tgt.ekf_x();
            sx = x[0]; sy = x[2]; sz = x[4];
            yaw = x[6]; omega = x[7]; rr = x[8]; ll = x[9]; hh = x[10];
            // 状态协方差对角线（σ²）：P 是否"真实"就看它与实际误差/残差是否匹配。
            const auto& ekf = tgt.ekf();
            if (ekf.P.rows() == 11)
                for (int i = 0; i < 11; ++i) pdiag[static_cast<std::size_t>(i)] = ekf.P(i, i);
            const auto nis_it = ekf.data.find("nis");
            if (nis_it != ekf.data.end()) nis = nis_it->second;
            const auto nis_pre_it = ekf.data.find("nis_pre");
            if (nis_pre_it != ekf.data.end()) nis_pre = nis_pre_it->second;
            if (tracker.state() == "tracking") ++tracked;

            // 重投影：sp_vision 自己的颜色标签（model 摆的板）→ 投影 → 与每个检测比
            const auto list = tgt.armor_xyza_list();
            std::vector<cv::Point2f> centers;
            for (const auto& xyza : list)
            {
                const Eigen::Vector3d xyz(xyza[0], xyza[1], xyza[2]);
                const auto px = solver.reproject_armor(xyz, xyza[3], tgt.armor_type, tgt.name);
                if (px.size() == 4)
                    centers.push_back(0.25f * (px[0] + px[1] + px[2] + px[3]));
            }
            std::vector<double> per_det;
            for (const Det& d : fr.dets)
            {
                const cv::Point2f c = 0.25f * (d.corners[0] + d.corners[1] + d.corners[2] +
                                               d.corners[3]);
                double best = 1e18;
                for (const auto& mc : centers) best = std::min(best, (double)cv::norm(mc - c));
                if (best < 1e17) per_det.push_back(best);
            }
            if (!per_det.empty())
            {
                // 被"钉住"的那块板不算：真机口径的 model_fit 也是这么算的。
                // sp_vision 没有硬钉，所以这里用"去掉最小那个"来近似。
                std::sort(per_det.begin(), per_det.end());
                detfit.insert(detfit.end(), per_det.begin(), per_det.end());
                med = per_det.front();
                mx = per_det.back();
                detfit_worst.push_back(mx);
            }
        }

        if (out)
        {
            out << std::fixed << std::setprecision(4) << t << ',' << tracker.state() << ','
                << fr.dets.size() << ',' << sx << ',' << sy << ',' << sz << ',' << yaw << ','
                << omega << ',' << rr << ',' << ll << ',' << hh << ',' << med << ',' << mx;
            for (int i = 0; i < 11; ++i) out << ',' << pdiag[static_cast<std::size_t>(i)];
            out << ',' << nis << ',' << nis_pre << '\n';
        }
        ++rows;
    }

    const auto pct = [](std::vector<double> v, double p) {
        if (v.empty()) return -1.0;
        std::sort(v.begin(), v.end());
        return v[static_cast<std::size_t>(p * (v.size() - 1))];
    };
    std::cout << "[spv] 帧数=" << rows << "  tracking=" << tracked;
    if (local_fit) std::cout << "  fit=" << fit_injected << "/" << fit_tried;
    std::cout << std::endl;
    std::cout << "[spv] 检测↔spv模型板心(px): n=" << detfit.size() << "  中位 " << pct(detfit, 0.5)
              << "  p90 " << pct(detfit, 0.9)
              << "   | 每帧最差的中位 " << pct(detfit_worst, 0.5) << std::endl;
    return 0;
}
