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
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--detections" && i + 1 < argc) det_path = argv[++i];
        else if (a == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--max-rows" && i + 1 < argc) max_rows = std::atoi(argv[++i]);
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
    const cv::Mat K = auto_aim::readMatFromYaml(intr["camera_matrix"]);
    const cv::Mat D = auto_aim::readMatFromYaml(intr["dist_coeffs"]);

    auto_aim::CameraPose hand_eye;   // 只为取手眼外参；姿态每帧覆盖
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
        out << "t,state,n_det,sx,sy,sz,yaw,omega,r,l,h,detfit_med,detfit_max\n";
    }

    const auto base_time = std::chrono::steady_clock::now();
    long long first_us = 0;
    bool have_first = false;
    int rows = 0;
    int tracked = 0;
    std::vector<double> detfit;
    std::vector<double> detfit_worst;

    for (const auto& [us, fr] : frames)
    {
        if (max_rows > 0 && rows >= max_rows) break;
        if (!have_first) { first_us = us; have_first = true; }
        const double t = static_cast<double>(us - first_us) * 1e-6;
        auto_aim::CameraPose pose;
        pose.yaw = yaw_sign * fr.yaw;
        pose.pitch = pitch_sign * fr.pitch;
        solver.set_R_gimbal2world(Eigen::Quaterniond(gimbalToWorld(pose)));

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
        if (!targets.empty())
        {
            const spv::Target& tgt = targets.front();
            const Eigen::VectorXd x = tgt.ekf_x();
            sx = x[0]; sy = x[2]; sz = x[4];
            yaw = x[6]; omega = x[7]; rr = x[8]; ll = x[9]; hh = x[10];
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
            out << std::fixed << std::setprecision(4) << t << ',' << tracker.state() << ','
                << fr.dets.size() << ',' << sx << ',' << sy << ',' << sz << ',' << yaw << ','
                << omega << ',' << rr << ',' << ll << ',' << hh << ',' << med << ',' << mx
                << '\n';
        ++rows;
    }

    const auto pct = [](std::vector<double> v, double p) {
        if (v.empty()) return -1.0;
        std::sort(v.begin(), v.end());
        return v[static_cast<std::size_t>(p * (v.size() - 1))];
    };
    std::cout << "[spv] 帧数=" << rows << "  tracking=" << tracked << std::endl;
    std::cout << "[spv] 检测↔spv模型板心(px): n=" << detfit.size() << "  中位 " << pct(detfit, 0.5)
              << "  p90 " << pct(detfit, 0.9)
              << "   | 每帧最差的中位 " << pct(detfit_worst, 0.5) << std::endl;
    return 0;
}
