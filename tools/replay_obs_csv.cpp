// 离线回放**实车检测流**（observation.csv / obs_rcv.csv），把同一份角点观测
// 重新喂进**和真机完全相同**的 Tracker，用来在没有实车的时候做 A/B。
//
// 为什么需要它：仿真器的角点是干净的、也没有 15° 支架，很多"实车才暴露"的
// 问题（模型中心被单帧 PnP 深度拖着跑、擦边板把 yaw 拽歪、选板抖动）在仿真里
// A/B 是负收益，却正是实车的主要误差源。有了这个工具，改一个 yaml 键就能
// 用**实车数据**对比，不必每次去车上跑。
//
// 输入（`observation.csv`，real_runner 的 observation_recorder 落盘格式）：
//   x,y,z,rx,ry,rz,tracker_x,tracker_y,tracker_z,cmd_yaw,cmd_pitch,time_us,
//   plate_id,state,u0,v0,u1,v1,u2,v2,u3,v3,source_time_us
//   · x,y,z,rx,ry,rz / tracker_* / plate_id / state 是录制时的输出，只用于对照；
//   · u0..v3 是四角像素（PnP 的输入顺序：左下、左上、右上、右下）；
//   · cmd_yaw/cmd_pitch 是**曝光时刻的下位机回传角**，回放时按 serial.yaml 的
//     yaw_sign/pitch_sign 变成相机姿态（与 real_runner 同一套口径）。
//
// 输出（--out，默认不写）：每帧一行，含 Tracker 状态 + 四块板的重投影像素，
// 便于用 python 统计"中心漂移 / ω 符号 / 重投影残差"。
//
// 用法：
//   replay_obs_csv <obs.csv> [--config-dir configs] [--out dump.csv]
//                  [--max-rows N] [--type small|large]
//                  [--set tracker.kalman.distance_noise_scale=1.0] ...

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config_loader.hpp"
#include "config/tracker_config_loader.hpp"
#include "Kalman/tracker.hpp"
#include "perception/pnp_solver.hpp"
#include "visualization/projection.hpp"
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

    bool parseCell(const std::vector<std::string>& row, int index, double& value)
    {
        if (index < 0 || index >= static_cast<int>(row.size())) return false;
        const std::string cell = trim(row[static_cast<std::size_t>(index)]);
        if (cell.empty() || cell == "nan" || cell == "None") return false;
        try
        {
            value = std::stod(cell);
        }
        catch (...)
        {
            return false;
        }
        return std::isfinite(value);
    }

    /// @brief `a.b.c=value` → 改 YAML 树。就是为了"改一个键做 A/B"而存在。
    void applyOverride(YAML::Node root, const std::string& spec)
    {
        const auto eq = spec.find('=');
        if (eq == std::string::npos)
        {
            std::cerr << "[replay] --set 需要 key=value：" << spec << std::endl;
            return;
        }
        const std::string path = spec.substr(0, eq);
        const std::string value = spec.substr(eq + 1);
        YAML::Node node = root;
        std::stringstream ss(path);
        std::string key;
        while (std::getline(ss, key, '.'))
        {
            if (!node[key]) node[key] = YAML::Node(YAML::NodeType::Map);
            // 注意：yaml-cpp 的 operator= 是"把 rhs 的内容赋给当前节点"，
            // 想沿树往下走必须用 reset()（重绑定句柄），否则整棵树会被改写掉。
            node.reset(node[key]);
        }
        try
        {
            node = std::stod(value);
        }
        catch (...)
        {
            if (value == "true" || value == "false") node = (value == "true");
            else node = value;
        }
    }

    auto_aim::Armor armorFromCorners(const std::array<cv::Point2f, 4>& corners, bool large)
    {
        // 顺序与 observation.csv / Points_2D 一致：左下、左上、右上、右下。
        auto_aim::Armor armor;
        armor.armor_type = large ? auto_aim::large : auto_aim::small;
        armor.number = -1;
        armor.solve_result = false;
        armor.left.bottom = corners[0];
        armor.left.top = corners[1];
        armor.right.top = corners[2];
        armor.right.bottom = corners[3];
        armor.left.center = 0.5f * (armor.left.top + armor.left.bottom);
        armor.right.center = 0.5f * (armor.right.top + armor.right.bottom);
        armor.Points_2D = {armor.left.bottom, armor.left.top, armor.right.top, armor.right.bottom};
        return armor;
    }
}

int main(int argc, char* argv[])
{
    std::string csv_path;
    std::string config_dir = "configs";
    std::string out_path;
    std::string armor_type = "small";
    std::string detections_path;
    int max_rows = 0;
    bool local_fit_enabled = false;
    int local_fit_period = 8;   // 每 N 帧跑一次拟合（真机 worker 是 30ms 周期，量级相当）
    std::vector<std::string> overrides;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--max-rows" && i + 1 < argc) max_rows = std::atoi(argv[++i]);
        else if (arg == "--type" && i + 1 < argc) armor_type = argv[++i];
        else if (arg == "--detections" && i + 1 < argc) detections_path = argv[++i];
        else if (arg == "--local-fit") local_fit_enabled = true;
        else if (arg == "--local-fit-period" && i + 1 < argc)
            local_fit_period = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--set" && i + 1 < argc) overrides.push_back(argv[++i]);
        else if (arg.rfind("--", 0) == 0)
        {
            std::cerr << "[replay] 未知参数：" << arg << std::endl;
            return 2;
        }
        else csv_path = arg;
    }
    if (csv_path.empty() && detections_path.empty())
    {
        std::cerr << "用法：replay_obs_csv <obs.csv> [--config-dir configs] [--out dump.csv]\n"
                     "                     [--max-rows N] [--type small|large]\n"
                     "                     [--local-fit]   # 接上局部拟合器（同步跑，不用 worker）\n"
                     "                     [--detections d.csv]  # 用\"全部检测\"流当输入\n"
                     "                                           # （ULTRA_VISION_DETECTIONS_CSV）\n"
                     "                     [--set tracker.kalman.<key>=<value>]...\n";
        return 2;
    }

    // ---- 配置：camera.yaml 内参 + serial.yaml 姿态符号 + tracker.yaml ----
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const YAML::Node camera_node = camera_file["camera"];
    const std::string camera_name = camera_node["name"].as<std::string>("galaxy");
    const YAML::Node intrinsic_node =
        (camera_name == "hikcamera") ? camera_file["hikcamera"] : camera_file["galaxy"];
    const cv::Mat camera_matrix = auto_aim::readMatFromYaml(intrinsic_node["camera_matrix"]);
    const cv::Mat dist_coeffs = auto_aim::readMatFromYaml(intrinsic_node["dist_coeffs"]);

    YAML::Node tracker_file = YAML::LoadFile(config_dir + "/tracker.yaml");
    for (const auto& spec : overrides) applyOverride(tracker_file, spec);

    auto_aim::TrackerConfig track_cfg = auto_aim::loadTrackerConfig(tracker_file);
    if (local_fit_enabled) track_cfg.local_fit.enabled = true;
    const auto_aim::PnpGeometry pnp_geometry = auto_aim::loadPnpGeometry(tracker_file);
    std::cout << "[replay] cfg hard_plate=" << track_cfg.hard_plate_constraint
              << " geometry_learn=" << track_cfg.geometry_learn
              << " joint_solve=" << track_cfg.joint_solve
              << " reanchor=" << track_cfg.reanchor_yaw_on_plate_switch
              << " nearest=" << track_cfg.nearest_plate_constraint
              << " pnp_yaw_trust=" << track_cfg.pnp_yaw_trust_deg
              << " dist_noise_scale=" << track_cfg.distance_noise_scale
              << " plate_lock=" << track_cfg.plate_lock_margin << std::endl;

    double yaw_sign = 1.0;
    double pitch_sign = 1.0;
    {
        const YAML::Node serial_file = YAML::LoadFile(config_dir + "/serial.yaml");
        const YAML::Node serial = serial_file["serial"] ? serial_file["serial"] : serial_file;
        yaw_sign = serial["yaw_sign"].as<double>(1.0);
        pitch_sign = serial["pitch_sign"].as<double>(1.0);
    }

    std::cout << "[replay] obs=" << csv_path << "  config=" << config_dir
              << "  fx=" << camera_matrix.at<double>(0, 0) << "  yaw_sign=" << yaw_sign
              << "  pitch_sign=" << pitch_sign << "  type=" << armor_type << std::endl;

    // ---- 可选：吃「全部检测」流（ULTRA_VISION_DETECTIONS_CSV）--------------------
    // 真机是把本帧**所有**检测喂给 tracker 的；只录一块板的 observation.csv 回放时会被迫
    // 退化成"每帧一块板"，于是"模型对**另外几块板**的解释误差"（也就是底盘构造对不对）
    // 根本量不出来 —— 实车实测：静止单板 model_fit 中位 0.9px，而同帧双板是 27px。
    // 给了 --detections 就以它为主输入（它自带云台角 + 全部检测 + 无检测帧）。
    struct DetFrame
    {
        double raw_yaw = 0.0;
        double raw_pitch = 0.0;
        std::vector<std::array<cv::Point2f, 4>> corners;
        std::vector<int> numbers;
    };
    std::map<long long, DetFrame> det_frames;
    if (!detections_path.empty())
    {
        std::ifstream din(detections_path);
        if (!din)
        {
            std::cerr << "[replay] 打不开 " << detections_path << std::endl;
            return 1;
        }
        std::string dh;
        std::getline(din, dh);
        const std::vector<std::string> dhead = splitCsv(dh);
        std::map<std::string, int> dcol;
        for (std::size_t i = 0; i < dhead.size(); ++i)
            dcol[trim(dhead[i])] = static_cast<int>(i);
        const auto dOf = [&](const std::string& n) {
            const auto it = dcol.find(n);
            return it == dcol.end() ? -1 : it->second;
        };
        const int i_t = dOf("time_us");
        const int i_y = dOf("cmd_yaw");
        const int i_p = dOf("cmd_pitch");
        const int i_num = dOf("armor_number");
        std::string dl;
        while (std::getline(din, dl))
        {
            if (dl.empty()) continue;
            const std::vector<std::string> r = splitCsv(dl);
            double tu = 0.0;
            double y = 0.0;
            double p = 0.0;
            if (!parseCell(r, i_t, tu)) continue;
            parseCell(r, i_y, y);
            parseCell(r, i_p, p);
            DetFrame& fr = det_frames[static_cast<long long>(std::llround(tu))];
            fr.raw_yaw = y;
            fr.raw_pitch = p;
            std::array<cv::Point2f, 4> c{};
            bool ok = true;
            for (int k = 0; k < 4; ++k)
            {
                double u = 0.0;
                double v = 0.0;
                if (!parseCell(r, dOf("u" + std::to_string(k)), u) ||
                    !parseCell(r, dOf("v" + std::to_string(k)), v))
                {
                    ok = false;
                    break;
                }
                c[static_cast<std::size_t>(k)] =
                    cv::Point2f(static_cast<float>(u), static_cast<float>(v));
            }
            if (ok)
            {
                double num = -1.0;
                parseCell(r, i_num, num);
                fr.corners.push_back(c);
                fr.numbers.push_back(static_cast<int>(num));
            }
        }
        std::cout << "[replay] detections=" << detections_path
                  << "  帧数=" << det_frames.size() << std::endl;
    }
    const bool use_det_stream = !det_frames.empty();

    std::ifstream in(csv_path);
    if (!use_det_stream && !in)
    {
        std::cerr << "[replay] 打不开 " << csv_path << std::endl;
        return 1;
    }

    std::string header_line;
    if (!use_det_stream && !std::getline(in, header_line))
    {
        std::cerr << "[replay] 空文件" << std::endl;
        return 1;
    }
    const std::vector<std::string> header = splitCsv(header_line);
    std::map<std::string, int> col;
    for (std::size_t i = 0; i < header.size(); ++i) col[trim(header[i])] = static_cast<int>(i);
    const auto colOf = [&](const std::string& name) {
        const auto it = col.find(name);
        return it == col.end() ? -1 : it->second;
    };
    const std::string time_name = colOf("time_us") >= 0 ? "time_us" : "t";
    const std::string yaw_name = colOf("cmd_yaw") >= 0 ? "cmd_yaw" : "yaw";
    const std::string pitch_name = colOf("cmd_pitch") >= 0 ? "cmd_pitch" : "pitch";
    std::array<int, 4> cu{};
    std::array<int, 4> cv_col{};
    for (int k = 0; k < 4; ++k)
    {
        cu[static_cast<std::size_t>(k)] = colOf("u" + std::to_string(k));
        cv_col[static_cast<std::size_t>(k)] = colOf("v" + std::to_string(k));
    }
    if (!use_det_stream && (colOf(time_name) < 0 || cu[0] < 0 || cv_col[0] < 0))
    {
        std::cerr << "[replay] 表头必须含 time_us 与 u0..v3（见 observation.csv）" << std::endl;
        return 1;
    }

    auto_aim::Tracker tracker(track_cfg);
    tracker.setCameraIntrinsics(camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1),
                                camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2));
    if (!dist_coeffs.empty()) tracker.setCameraDistortion(dist_coeffs);

#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
    // 局部拟合器：与 real_runner 同一套配置口径，但**同步**跑（不起 worker），
    // 这样 A/B 是确定性的、也不受这台机器 CPU 快慢影响。
    std::unique_ptr<auto_aim::LocalTrajectoryFitter> fitter;
    if (local_fit_enabled)
    {
        const auto env_d = [](const char* n, double d) {
            const char* v = std::getenv(n);
            return v != nullptr ? std::atof(v) : d;
        };
        auto_aim::LocalTrajectoryFitConfig fcfg;
        fcfg.K = {camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1),
                  camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2)};
        fcfg.max_rmse_deg = env_d("ULTRA_VISION_LOCAL_FIT_MAX_RMSE_DEG", 0.4);
        fcfg.lightbar_len = env_d("ULTRA_VISION_LOCAL_FIT_L", 0.060);
        fcfg.plate_width = env_d("ULTRA_VISION_LOCAL_FIT_W", 0.135);
        fcfg.gimbal_pitch = 0.0;
        fcfg.center_init = {{0.0, 0.0, 3.0}};
        fitter = std::make_unique<auto_aim::LocalTrajectoryFitter>(fcfg);
        const double alpha0 = env_d("ULTRA_VISION_LOCAL_FIT_ALPHA_DEG", 15.0) * M_PI / 180.0;
        fitter->seedSlow(alpha0, fcfg.R_prior);
        auto* raw = fitter.get();
        tracker.setLocalFitter([raw](const std::vector<auto_aim::LightBarObservation>& obs,
                                     auto_aim::LocalFitResult& out) {
            return raw->fitWithSlow(obs, out);
        });
        std::cout << "[replay] local-fit ON (alpha="
                  << env_d("ULTRA_VISION_LOCAL_FIT_ALPHA_DEG", 15.0) << "deg, use_yaw="
                  << track_cfg.local_fit.use_yaw << ", use_fit_radius="
                  << track_cfg.local_fit.use_fit_radius << ")" << std::endl;
    }
#else
    if (local_fit_enabled)
        std::cerr << "[replay] 本次构建没有局部拟合器（未找到 Ceres），--local-fit 无效" << std::endl;
#endif

    std::ofstream out;
    if (!out_path.empty())
    {
        out.open(out_path, std::ios::out | std::ios::trunc);
        out << "t,state,plate_id,accepted,cx,cy,cz,yaw,omega,fit_px,"
               "obs_plate,obs_px,obs_py,model_px,model_py,"
               "u0,v0,u1,v1,u2,v2,u3,v3";
        for (int p = 0; p < 4; ++p)
            for (int k = 0; k < 4; ++k)
                out << ",q" << p << "x" << k << ",q" << p << "y" << k;
        // 诊断列**必须在最后**，且与写入顺序一致（错位过一次，教训）。
        out << ",fit_yaw_raw_deg,ekf_yaw_deg,fit_omega";
        // EKF 状态协方差对角线（诊断"P 是否真实"）：0=x 1=vx 2=y 3=vy 4=sin
        // 5=cos 6=ω 7=z 8=vz 9=l 10=h。
        for (int i = 0; i < 11; ++i) out << ",P" << i;
        out << '\n';
    }

    const bool large = (armor_type == "large");
    std::string line;
    int rows = 0;
    int tracked = 0;
    int accepted_frames = 0;
    double first_time = 0.0;
    bool have_first = false;
    std::vector<double> cz_values;
    std::vector<double> omega_values;
    std::vector<double> fit_values;
    std::vector<double> det_fit_values;   // 每块检测到"最近模型板心"的像素距离
    int local_fit_counter = 0;
    std::array<int, 6> yaw_res_hist{{0, 0, 0, 0, 0, 0}};   // |残差| 分档：<15/45/75/105/135/180°

    std::map<long long, DetFrame>::const_iterator dit = det_frames.begin();
    while (true)
    {
        if (max_rows > 0 && rows >= max_rows) break;
        std::vector<std::array<cv::Point2f, 4>> frame_dets;
        std::vector<int> frame_numbers;
        double time_us = 0.0;
        double raw_yaw = 0.0;
        double raw_pitch = 0.0;
        bool has_yaw = false;
        bool has_pitch = false;
        if (use_det_stream)
        {
            if (dit == det_frames.end()) break;
            time_us = static_cast<double>(dit->first);
            raw_yaw = dit->second.raw_yaw;
            raw_pitch = dit->second.raw_pitch;
            has_yaw = has_pitch = true;
            frame_dets = dit->second.corners;
            frame_numbers = dit->second.numbers;
            ++dit;
        }
        else
        {
            if (!std::getline(in, line)) break;
            if (line.empty()) continue;
            const std::vector<std::string> row = splitCsv(line);
            if (!parseCell(row, colOf(time_name), time_us)) continue;
            has_yaw = parseCell(row, colOf(yaw_name), raw_yaw);
            has_pitch = parseCell(row, colOf(pitch_name), raw_pitch);
            std::array<cv::Point2f, 4> c{};
            bool ok = true;
            for (int k = 0; k < 4; ++k)
            {
                double u = 0.0;
                double v = 0.0;
                if (!parseCell(row, cu[static_cast<std::size_t>(k)], u) ||
                    !parseCell(row, cv_col[static_cast<std::size_t>(k)], v))
                {
                    ok = false;
                    break;
                }
                c[static_cast<std::size_t>(k)] =
                    cv::Point2f(static_cast<float>(u), static_cast<float>(v));
            }
            if (!ok)
            {
                ++rows;
                continue;
            }
            frame_dets.push_back(c);
            double num = -1.0;
            parseCell(row, colOf("armor_number"), num);
            frame_numbers.push_back(static_cast<int>(num));
        }
        if (!have_first)
        {
            first_time = time_us;
            have_first = true;
        }
        const double t = (time_us - first_time) * 1e-6;

        // 本帧**全部**检测 → armor（与 real_runner 同口径：都做 PnP）。
        std::vector<auto_aim::Armor> armors;
        for (std::size_t i = 0; i < frame_dets.size(); ++i)
        {
            auto_aim::Armor a = armorFromCorners(frame_dets[i], large);
            if (i < frame_numbers.size()) a.number = frame_numbers[i];
            a.solve_result = auto_aim::solveArmorPnP(a, camera_matrix, dist_coeffs, pnp_geometry);
            if (a.solve_result) armors.push_back(a);
        }
        const std::array<cv::Point2f, 4> corners = frame_dets.empty()
            ? std::array<cv::Point2f, 4>{}
            : frame_dets.front();

        // ---- 相机姿态（real_runner 口径：内部角 = sign × 回传角）----
        auto_aim::CameraPose pose;
        pose.yaw = yaw_sign * (has_yaw ? raw_yaw : 0.0);
        pose.pitch = pitch_sign * (has_pitch ? raw_pitch : 0.0);
        pose.valid = true;
        pose.timestamp = t;
        pose.timestamp_valid = true;
        tracker.setCameraPose(pose);

#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
        bool ran_fit = false;
        if (fitter)
        {
            // 与 real_runner 同序：先把**本帧全部检测**推进拟合缓冲（在关联之前），
            // 再跑一次拟合，最后 update + apply。同步调用 → 确定性、与 CPU 快慢无关。
            auto fcfg = fitter->config();
            fcfg.h_prior = tracker.chassisGeometry().height(0);
            fcfg.use_rcv = true;
            fcfg.rcv = tracker.fitterRcv();
            fitter->setConfig(fcfg);
            // A/B：默认把**本帧全部检测**都推入（历史行为）。设 SPV_PUSH_FIRST=1 时
            // 只推第一块 —— 用来验证"同帧多板被串进同一条弧"到底伤不伤拟合质量。
            const bool push_first = std::getenv("SPV_PUSH_FIRST") != nullptr;
            for (const auto& a : armors)
            {
                tracker.pushArmorLightBars(a, t, tracker.plateIdFor(a));
                if (push_first) break;
            }
            // 诊断（ULTRA_VISION_LOCAL_FIT_DIAG=1）：把"喂给拟合器的前提"打出来 ——
            // h 先验（来自底盘几何慢层）/ 本帧推入几块板 / 几何的四个板距。
            // 拟合 rmse 大时先看这里：h 先验给错（σ=5cm）会把残差顶到几十 px。
            if (std::getenv("ULTRA_VISION_LOCAL_FIT_DIAG") != nullptr && local_fit_counter % 200 == 0)
            {
                std::cout << "[diag] frame=" << local_fit_counter
                          << " h_prior=" << fcfg.h_prior << " alpha=" << fcfg.alpha_prior
                          << " W=" << fcfg.plate_width << " L=" << fcfg.lightbar_len
                          << " n_push=" << armors.size() << " use_rcv=" << fcfg.use_rcv
                          << " Rgeo=" << tracker.getArmorRadius() << "/"
                          << tracker.chassisGeometry().distance(1) << "/"
                          << tracker.chassisGeometry().distance(2) << "/"
                          << tracker.chassisGeometry().distance(3) << std::endl;
            }
            if (local_fit_counter++ % local_fit_period == 0) ran_fit = tracker.runLocalFitNow();
        }
#endif
        tracker.update(armors, t);
#ifdef ULTRA_VISION_HAVE_LOCAL_FIT
        if (fitter) tracker.applyLatestLocalFit();
        // 相位残差直方图：判断"板号/相位表不一致"（残差贴 ±90°/180°）还是
        // "常值偏置"（残差几十度）。只在真正跑过拟合的帧采样，避免重复计数。
        if (ran_fit)
        {
            const double deg = std::abs(tracker.localFitYawResidual()) * 180.0 / M_PI;
            if (deg < 15.0) ++yaw_res_hist[0];
            else if (deg < 45.0) ++yaw_res_hist[1];
            else if (deg < 75.0) ++yaw_res_hist[2];
            else if (deg < 105.0) ++yaw_res_hist[3];
            else if (deg < 135.0) ++yaw_res_hist[4];
            else ++yaw_res_hist[5];
        }
#endif
        const bool accepted = tracker.hasObservation() &&
            std::abs(tracker.lastObservationTime() - t) < 1e-9;
        if (tracker.getState() == auto_aim::TrackerState::TRACKING) ++tracked;
        if (accepted) ++accepted_frames;

        // ---- 模型快照（相机系）→ 画到图上的那四个框 ----
        bool have_model = false;
        double cx = 0.0;
        double cy = 0.0;
        double cz = 0.0;
        double fit_px = -1.0;
        double obs_px = 0.0;
        double obs_py = 0.0;
        double model_px = 0.0;
        double model_py = 0.0;
        // 16 个模型角点：**必须**在声明时就填满。只在 have_model 分支里 assign
        // 会让"丢目标/快照无效"的帧在落盘时越界访问空 vector（实测段错误，
        // 2026-10-11 修）。
        std::vector<cv::Point2f> all_px(16, cv::Point2f(-1.0f, -1.0f));
        const int obs_plate = tracker.getLastObservedPlateId();
        if (tracker.getState() != auto_aim::TrackerState::LOST)
        {
            const auto snapshot = tracker.modelAt(0.0, pose);
            if (snapshot.valid)
            {
                have_model = true;
                cx = snapshot.center_camera[0];
                cy = snapshot.center_camera[1];
                cz = snapshot.center_camera[2];
                all_px.assign(16, cv::Point2f(-1.0f, -1.0f));
                for (int p = 0; p < 4; ++p)
                {
                    const auto& sp = snapshot.poses[static_cast<std::size_t>(p)];
                    const cv::Mat center_mat =
                        (cv::Mat_<double>(3, 1) << sp.center[0], sp.center[1], sp.center[2]);
                    const auto pixels = auto_aim::projectArmorCameraFrame(
                        center_mat, sp.axes.tangent, sp.axes.bar, snapshot.geometry.halfWidth(p),
                        snapshot.geometry.half_height, camera_matrix, dist_coeffs);
                    for (int k = 0; k < 4 && k < static_cast<int>(pixels.size()); ++k)
                        all_px[static_cast<std::size_t>(p * 4 + k)] =
                            pixels[static_cast<std::size_t>(k)];
                    if (p == obs_plate)
                    {
                        obs_px = sp.center[0];
                        obs_py = sp.center[1];
                    }
                }
                // 残差：模型画出的**观测那块板**的四角 vs 本帧观测四角（逐角匹配）。
                double sx = 0.0;
                double sy = 0.0;
                double sum = 0.0;
                for (int k = 0; k < 4; ++k)
                {
                    sum += cv::norm(all_px[static_cast<std::size_t>(obs_plate * 4 + k)] -
                                    corners[static_cast<std::size_t>(k)]);
                    sx += corners[static_cast<std::size_t>(k)].x;
                    sy += corners[static_cast<std::size_t>(k)].y;
                }
                fit_px = sum / 4.0;
                model_px = 0.25 * (all_px[static_cast<std::size_t>(obs_plate * 4 + 0)].x +
                                   all_px[static_cast<std::size_t>(obs_plate * 4 + 1)].x +
                                   all_px[static_cast<std::size_t>(obs_plate * 4 + 2)].x +
                                   all_px[static_cast<std::size_t>(obs_plate * 4 + 3)].x);
                model_py = 0.25 * (all_px[static_cast<std::size_t>(obs_plate * 4 + 0)].y +
                                   all_px[static_cast<std::size_t>(obs_plate * 4 + 1)].y +
                                   all_px[static_cast<std::size_t>(obs_plate * 4 + 2)].y +
                                   all_px[static_cast<std::size_t>(obs_plate * 4 + 3)].y);
                obs_px = sx / 4.0;
                obs_py = sy / 4.0;
            }
        }

        if (have_model)
        {
            cz_values.push_back(cz);
            omega_values.push_back(tracker.getOmega());
            if (fit_px >= 0.0) fit_values.push_back(fit_px);
            // ---- "非观测板"指标：每块检测 ↔ 最近的模型板心 --------------------
            // 被钉住那块板按定义永远重合，所以"底盘构造对不对"只能看这个量
            // （与 real_runner 状态行的 model_fit 同义）。同帧只有一块板时它≈0；
            // 同帧两块板时它才真正暴露构造误差。
            for (const auto& a : armors)
            {
                const cv::Point2f dc =
                    0.25f * (a.left.top + a.left.bottom + a.right.top + a.right.bottom);
                double best = 1e18;
                for (int p = 0; p < 4; ++p)
                {
                    cv::Point2f ctr(0.0f, 0.0f);
                    int n = 0;
                    for (int k = 0; k < 4; ++k)
                    {
                        const cv::Point2f& q = all_px[static_cast<std::size_t>(p * 4 + k)];
                        if (q.x < 0.0f) continue;
                        ctr += q;
                        ++n;
                    }
                    if (n == 0) continue;
                    ctr *= 1.0f / static_cast<float>(n);
                    best = std::min(best, static_cast<double>(cv::norm(ctr - dc)));
                }
                if (best < 1e17) det_fit_values.push_back(best);
            }
        }

        if (out)
        {
            out << std::fixed << std::setprecision(4) << t << ','
                << static_cast<int>(tracker.getState()) << ',' << obs_plate << ','
                << (accepted ? 1 : 0) << ',' << cx << ',' << cy << ',' << cz << ','
                << tracker.getYaw() << ',' << tracker.getOmega() << ',' << fit_px << ','
                << obs_plate << ',' << obs_px << ',' << obs_py << ',' << model_px << ','
                << model_py;
            for (int k = 0; k < 4; ++k)
            {
                out << ',' << corners[static_cast<std::size_t>(k)].x << ','
                    << corners[static_cast<std::size_t>(k)].y;
            }
            for (int p = 0; p < 16; ++p)
                out << ',' << all_px[static_cast<std::size_t>(p)].x << ','
                    << all_px[static_cast<std::size_t>(p)].y;
            // 诊断两列：拟合器**原始**输出的 yaw（相机系，未换算）与 EKF 的 yaw。
            // 两者一起看就能判断"拟合 yaw ↔ EKF yaw"到底是镜像、常值偏置还是差 90°。
            double fit_yaw_raw = 0.0;
            double fit_omega = 0.0;
            bool have_fit = false;
            if (auto lf = tracker.latestLocalFit(); lf && lf->valid)
            {
                fit_yaw_raw = lf->yaw * 180.0 / M_PI;
                fit_omega = lf->omega;
                have_fit = true;
            }
            out << ',' << (have_fit ? fit_yaw_raw : 0.0) << ','
                << (have_fit ? tracker.getYaw() * 180.0 / M_PI : 0.0) << ','
                << (have_fit ? fit_omega : 0.0);
            for (int i = 0; i < 11; ++i) out << ',' << tracker.ekfCovarianceDiag(i);
            out << '\n';
        }
        ++rows;
    }

    const auto median = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::cout << "[replay] rows=" << rows << "  tracking=" << tracked
              << "  accepted=" << accepted_frames << std::endl;
    std::cout << "[replay] model samples=" << cz_values.size()
              << "  cz median=" << median(cz_values)
              << "  |omega| median=" << median(omega_values)
              << "  fit_px median=" << median(fit_values);
    if (!det_fit_values.empty())
    {
        std::vector<double> v = det_fit_values;
        std::sort(v.begin(), v.end());
        std::cout << "\n[replay] 全部检测↔模型板心(px): n=" << v.size()
                  << "  中位 " << v[v.size() / 2] << "  p90 " << v[(v.size() * 9) / 10]
                  << "  （这才是底盘构造对不对的度量；被钉那块板不算）";
    }
    if (cz_values.size() > 1)
    {
        const auto mm = std::minmax_element(cz_values.begin(), cz_values.end());
        std::cout << "  cz range=[" << *mm.first << ", " << *mm.second << "]";
    }
    std::cout << std::endl;
    if (local_fit_enabled)
    {
        std::cout << "[replay] local-fit: attempts=" << tracker.localFitAttempts()
                  << " valids=" << tracker.localFitValids()
                  << " applied=" << tracker.localFitApplied()
                  << " geometry=" << tracker.localFitApplyGeometry()
                  << " ysnap=" << tracker.localFitYawSnaps()
                  // ygate=被 yaw 质量门拦下的次数（rmse>yaw_max_rmse_px 或 相位跨度不足）
                  << " ygate=" << tracker.localFitYawGated()
                  << " yexact=" << tracker.localFitYawExact()
                  << " yspan=" << tracker.localFitPhaseSpanDeg()
                  << " ogate=" << tracker.localFitOmegaGated()
                  << " lastR=" << tracker.localFitLastR()
                  << " lastRmse=" << tracker.localFitLastRmse()
                  << " | Rgeo=" << tracker.chassisGeometry().distance(0) << "/"
                  << tracker.chassisGeometry().distance(1) << "/"
                  << tracker.chassisGeometry().distance(2) << "/"
                  << tracker.chassisGeometry().distance(3) << std::endl;
        // 注意：对四块板 90° 对称的车体，"残差落在 90° 的整数倍上"是**正常歧义**
        // （底盘 yaw 只在模 90° 下可定义），会走吸附路径；真正要看的是
        // "90° 倍数之外还剩多少"——那部分才是拟合器与 EKF 的真实相位差（由
        // yaw_bias 学习器吸收）。所以 75-105/135-180 这两档偏大不代表出错。
        std::cout << "[replay] yaw 残差直方图 |fit.yaw-EKF.yaw| (deg): <15:"
                  << yaw_res_hist[0] << " 15-45:" << yaw_res_hist[1]
                  << " 45-75:" << yaw_res_hist[2] << " 75-105:" << yaw_res_hist[3]
                  << " 105-135:" << yaw_res_hist[4] << " 135-180:" << yaw_res_hist[5]
                  << "  | exact=" << tracker.localFitYawExact()
                  << " snap=" << tracker.localFitYawSnaps() << std::endl;
    }
    return 0;
}
