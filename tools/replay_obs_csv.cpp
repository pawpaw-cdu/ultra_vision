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
    int max_rows = 0;
    std::vector<std::string> overrides;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--max-rows" && i + 1 < argc) max_rows = std::atoi(argv[++i]);
        else if (arg == "--type" && i + 1 < argc) armor_type = argv[++i];
        else if (arg == "--set" && i + 1 < argc) overrides.push_back(argv[++i]);
        else if (arg.rfind("--", 0) == 0)
        {
            std::cerr << "[replay] 未知参数：" << arg << std::endl;
            return 2;
        }
        else csv_path = arg;
    }
    if (csv_path.empty())
    {
        std::cerr << "用法：replay_obs_csv <obs.csv> [--config-dir configs] [--out dump.csv]\n"
                     "                     [--max-rows N] [--type small|large]\n"
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

    const auto_aim::TrackerConfig track_cfg = auto_aim::loadTrackerConfig(tracker_file);
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

    std::ifstream in(csv_path);
    if (!in)
    {
        std::cerr << "[replay] 打不开 " << csv_path << std::endl;
        return 1;
    }

    std::string header_line;
    if (!std::getline(in, header_line))
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
    if (colOf(time_name) < 0 || cu[0] < 0 || cv_col[0] < 0)
    {
        std::cerr << "[replay] 表头必须含 time_us 与 u0..v3（见 observation.csv）" << std::endl;
        return 1;
    }

    auto_aim::Tracker tracker(track_cfg);
    tracker.setCameraIntrinsics(camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1),
                                camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2));
    if (!dist_coeffs.empty()) tracker.setCameraDistortion(dist_coeffs);

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

    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        if (max_rows > 0 && rows >= max_rows) break;
        const std::vector<std::string> row = splitCsv(line);

        double time_us = 0.0;
        if (!parseCell(row, colOf(time_name), time_us)) continue;
        if (!have_first)
        {
            first_time = time_us;
            have_first = true;
        }
        const double t = (time_us - first_time) * 1e-6;

        std::array<cv::Point2f, 4> corners{};
        bool corners_ok = true;
        for (int k = 0; k < 4; ++k)
        {
            double u = 0.0;
            double v = 0.0;
            if (!parseCell(row, cu[static_cast<std::size_t>(k)], u) ||
                !parseCell(row, cv_col[static_cast<std::size_t>(k)], v))
            {
                corners_ok = false;
                break;
            }
            corners[static_cast<std::size_t>(k)] =
                cv::Point2f(static_cast<float>(u), static_cast<float>(v));
        }
        if (!corners_ok)
        {
            ++rows;
            continue;
        }

        auto_aim::Armor armor = armorFromCorners(corners, large);
        // 板面数字（分类器给的机器人 ID）：老录制没有这一列 → 保持 -1。
        double number = -1.0;
        if (parseCell(row, colOf("armor_number"), number))
            armor.number = static_cast<int>(number);
        armor.solve_result =
            auto_aim::solveArmorPnP(armor, camera_matrix, dist_coeffs, pnp_geometry);

        // ---- 相机姿态（real_runner 口径：内部角 = sign × 回传角）----
        auto_aim::CameraPose pose;
        double raw_yaw = 0.0;
        double raw_pitch = 0.0;
        const bool has_yaw = parseCell(row, colOf(yaw_name), raw_yaw);
        const bool has_pitch = parseCell(row, colOf(pitch_name), raw_pitch);
        pose.yaw = yaw_sign * (has_yaw ? raw_yaw : 0.0);
        pose.pitch = pitch_sign * (has_pitch ? raw_pitch : 0.0);
        pose.valid = true;
        pose.timestamp = t;
        pose.timestamp_valid = true;
        tracker.setCameraPose(pose);

        const std::vector<auto_aim::Armor> armors =
            armor.solve_result ? std::vector<auto_aim::Armor>{armor}
                               : std::vector<auto_aim::Armor>{};
        tracker.update(armors, t);
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
        std::vector<cv::Point2f> all_px;
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
    if (cz_values.size() > 1)
    {
        const auto mm = std::minmax_element(cz_values.begin(), cz_values.end());
        std::cout << "  cz range=[" << *mm.first << ", " << *mm.second << "]";
    }
    std::cout << std::endl;
    return 0;
}
