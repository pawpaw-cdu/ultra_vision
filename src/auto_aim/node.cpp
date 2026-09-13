#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "common/types.hpp"
#include "perception/detector.hpp"
#include "tools/visualizer.hpp"
#include "perception/pnp_solver.hpp"
#include "Kalman/tracker.hpp"
#include "io/camera/GalaxyCamera.hpp"

#ifndef ULTRA_VISION_CONFIG_DIR
#define ULTRA_VISION_CONFIG_DIR "configs"
#endif

// 辅助函数：从 YAML 节点读取矩阵
cv::Mat readMatFromYaml(const YAML::Node & node) {
    int rows = node["rows"].as<int>();
    int cols = node["cols"].as<int>();
    std::vector<double> data = node["data"].as<std::vector<double>>();
    cv::Mat mat(rows, cols, CV_64F);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat.at<double>(i, j) = data[i * cols + j];
    return mat;
}

int main(int argc, char* argv[]) {
    std::string config_dir = ULTRA_VISION_CONFIG_DIR;
    if (argc > 1) {
        config_dir = argv[1];
    }
    if (const char* env = std::getenv("ULTRA_VISION_CONFIG_DIR")) {
        config_dir = env;
    }
    auto config_path = [&](const std::string& name) {
        return config_dir + "/" + name;
    };

    // ------------------- 1. 加载配置文件 -------------------
    YAML::Node camera_cfg = YAML::LoadFile(config_path("camera.yaml"));
    YAML::Node detector_cfg = YAML::LoadFile(config_path("detector.yaml"));
    YAML::Node tracker_cfg = YAML::LoadFile(config_path("tracker.yaml"));

    // 相机参数
     std::string camera_name = camera_cfg["camera"]["name"].as<std::string>();
     int device_index = camera_cfg["camera"]["device_index"].as<int>();
     std::string serial_number = camera_cfg["camera"]["serial_number"].as<std::string>();
     int camera_timeout = camera_cfg["camera"]["timeout_ms"].as<int>();
     cv::Mat camera_matrix;
     cv::Mat dist_coeffs;
     if(camera_name == "galaxy"){
         camera_matrix = readMatFromYaml(camera_cfg["galaxy"]["camera_matrix"]);
         dist_coeffs = readMatFromYaml(camera_cfg["galaxy"]["dist_coeffs"]);
     }else if(camera_name == "hikcamera"){
         camera_matrix = readMatFromYaml(camera_cfg["hikcamera"]["camera_matrix"]);
         dist_coeffs = readMatFromYaml(camera_cfg["hikcamera"]["dist_coeffs"]);
     }
        
    
    // 检测器参数
    auto_aim::DetectorConfig det_cfg;
    det_cfg.enemy_color = detector_cfg["detector"]["enemy_color"].as<int>();
    det_cfg.binary_threshold = detector_cfg["detector"]["binary_threshold"].as<int>();
    auto light = detector_cfg["detector"]["light"];
    det_cfg.light_min_ratio = light["min_ratio"].as<double>();
    det_cfg.light_max_ratio = light["max_ratio"].as<double>();
    det_cfg.light_max_angle = light["max_angle"].as<double>();
    det_cfg.light_min_contour_points = light["min_contour_points"].as<int>();
    auto armor = detector_cfg["detector"]["armor"];
    det_cfg.armor_height_ratio_min = armor["height_ratio_min"].as<double>();
    det_cfg.armor_height_ratio_max = armor["height_ratio_max"].as<double>();
    det_cfg.armor_angle_diff_max = armor["angle_diff_max"].as<double>();
    det_cfg.armor_width_to_height_min = armor["width_to_height_min"].as<double>();
    det_cfg.armor_width_to_height_max = armor["width_to_height_max"].as<double>();
    det_cfg.armor_large_ratio_thresh = armor["large_armor_ratio_thresh"].as<double>();

    auto color = detector_cfg["detector"]["color"];
    det_cfg.color_use_detect = color["use_color_detect"].as<bool>();
    std::string method = color["method"].as<std::string>();
    det_cfg.use_hsv = (method == "hsv");

    if (det_cfg.use_hsv) {
        auto hsv = color["hsv"];
        det_cfg.hue_red_low1   = hsv["red_low1"].as<int>();
        det_cfg.hue_red_high1  = hsv["red_high1"].as<int>();
        det_cfg.hue_red_low2   = hsv["red_low2"].as<int>();
        det_cfg.hue_red_high2  = hsv["red_high2"].as<int>();
        det_cfg.hue_blue_low   = hsv["blue_low"].as<int>();
        det_cfg.hue_blue_high  = hsv["blue_high"].as<int>();
        det_cfg.sat_min        = hsv["sat_min"].as<int>();
        det_cfg.val_min        = hsv["val_min"].as<int>();
        det_cfg.color_area_ratio = hsv["area_ratio"].as<double>();
    } else {
        auto bgr = color["bgr"];
        det_cfg.color_red_threshold  = bgr["red_threshold"].as<double>();
        det_cfg.color_blue_threshold = bgr["blue_threshold"].as<double>();
    }

    // 跟踪器参数
    auto_aim::TrackerConfig track_cfg;
    track_cfg.min_detect_frames = tracker_cfg["tracker"]["min_detect_frames"].as<int>();
    track_cfg.max_lost_frames = tracker_cfg["tracker"]["max_lost_frames"].as<int>();
    track_cfg.reacquire_frames =
        tracker_cfg["tracker"]["reacquire_frames"].as<int>(track_cfg.reacquire_frames);
    track_cfg.max_match_distance = tracker_cfg["tracker"]["max_match_distance"].as<double>();
    track_cfg.reacquire_center_error =
        tracker_cfg["tracker"]["reacquire_center_error"].as<double>(
            track_cfg.reacquire_center_error);
    track_cfg.reacquire_max_dt =
        tracker_cfg["tracker"]["reacquire_max_dt"].as<double>(track_cfg.reacquire_max_dt);
    track_cfg.max_camera_pose_dt =
        tracker_cfg["tracker"]["max_camera_pose_dt"].as<double>(
            track_cfg.max_camera_pose_dt);
    auto kalman = tracker_cfg["tracker"]["kalman"];
    track_cfg.state_dim = kalman["state_dim"].as<int>();
    track_cfg.measure_dim = kalman["measure_dim"].as<int>();
    track_cfg.process_noise_pos = kalman["process_noise_pos"].as<double>();
    track_cfg.process_noise_vel = kalman["process_noise_vel"].as<double>();
    track_cfg.measure_noise = kalman["measure_noise"].as<double>();
    track_cfg.init_error_cov = kalman["init_error_cov"].as<double>();
    track_cfg.default_dt = kalman["default_dt"].as<double>();
    track_cfg.armor_radius = kalman["armor_radius"].as<double>(0.21);
    track_cfg.y_noise_mult = kalman["y_noise_mult"].as<double>(30.0);
    track_cfg.p_cov_min = kalman["p_cov_min"].as<double>(1e-3);
    track_cfg.angle_measure_noise = kalman["angle_measure_noise"].as<double>(0.04);
    track_cfg.bearing_measure_noise = kalman["bearing_measure_noise"].as<double>(0.0025);
    track_cfg.pitch_measure_noise = kalman["pitch_measure_noise"].as<double>(0.005);
    track_cfg.range_noise_distance_scale =
        kalman["range_noise_distance_scale"].as<double>(0.1);
    track_cfg.ypd_nis_threshold = kalman["ypd_nis_threshold"].as<double>(9.488);
    track_cfg.max_angle_error = kalman["max_angle_error"].as<double>(0.70);
    track_cfg.angle_match_weight = kalman["angle_match_weight"].as<double>(0.08);
    track_cfg.position_nis_threshold = kalman["position_nis_threshold"].as<double>(7.815);
    track_cfg.angle_nis_threshold = kalman["angle_nis_threshold"].as<double>(3.841);

    // ------------------- 2. 初始化大恒相机 -------------------
    rm_ultra::GalaxyCamera camera;
    bool init_ok;
    if (serial_number.empty())
        init_ok = camera.init("", device_index);
    else
        init_ok = camera.init(serial_number, device_index);
    if (!init_ok) {
        std::cerr << "Failed to init Galaxy camera!" << std::endl;
        return -1;
    }
    std::cout << "Galaxy camera initialized successfully." << std::endl;

    // ------------------- 3. 设置 PnP 全局内参 -------------------
    // 假设 pnp_solver 中的全局变量 CAMERA_MATRIX 和 DIST_COEFFS 已声明为 extern
    // 此处需要更新它们（或者修改 solveArmorPnP 直接传入，这里简单演示直接修改全局变量）
    auto_aim::CAMERA_MATRIX = camera_matrix;
    auto_aim::DIST_COEFFS = dist_coeffs;

    // ------------------- 4. 多目标跟踪器列表 -------------------
    std::vector<std::unique_ptr<auto_aim::Tracker>> trackers;

    auto distance3D = [](const cv::Mat& t1, const cv::Mat& t2) {
        double dx = t1.at<double>(0) - t2.at<double>(0);
        double dy = t1.at<double>(1) - t2.at<double>(1);
        double dz = t1.at<double>(2) - t2.at<double>(2);
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    cv::Mat img;
    int key = 1;
    // cv::VideoCapture cap("test_vid_pic/auto_aim.mp4");
    while (key) {
        //cap >> img;
        cv::resize(img, img ,cv::Size(400,400));
        // 获取一帧图像
        if (!camera.getImage(img, camera_timeout)) {
            std::cerr << "Failed to get image from camera" << std::endl;
            break;
        }
        // if (img.empty()) {
        //     std::cerr << "Empty frame captured" << std::endl;
        //     return -1;
        // }

        // ------------------- 检测器 -------------------
        auto_aim::Detector detector(img, det_cfg);
        cv::Mat gray, binary,dilated;
        detector.gray_img(detector.clone_img, gray);
        detector.binary_img(gray, binary, det_cfg.binary_threshold);
        detector.open_close_img(binary,dilated);

        std::vector<std::vector<cv::Point>> contours;
        detector.find_contours(dilated, contours);
        // for(auto contour:contours){
        //     tools::draw_points(img,contour,cv::Scalar(0,255,0));
        // }

        std::vector<auto_aim::Light> lights;
        detector.find_lights(contours, lights, img, det_cfg.enemy_color);

        if(lights.empty()){
            std::cout <<"Not Found Light" << std::endl;
        }

        for(auto light:lights){
            std::cout << light.tilt_angle << std::endl;
        }

        std::vector<auto_aim::Armor> armors;
        detector.find_armors(lights, armors);
        std::cout << "=== Debug: armors.size() = " << armors.size() << std::endl;

        // PnP 解算
        for (auto& armor : armors) {
            bool ok = auto_aim::solveArmorPnP(armor, auto_aim::CAMERA_MATRIX, auto_aim::DIST_COEFFS);
            armor.solve_result = ok;
            std::cout << "PnP success: " << ok << ", distance = " << cv::norm(armor.tvec) << std::endl;
        }

        // ------------------- 跟踪器 -------------------
        auto now = std::chrono::steady_clock::now();
        double timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();

        std::vector<auto_aim::Armor> predicted_armors;
        for (auto& t : trackers) {
            if (t->getState() != auto_aim::TrackerState::LOST)
                predicted_armors.push_back(t->predict(timestamp));
            else
                predicted_armors.emplace_back();
        }

        std::vector<bool> tracker_matched(trackers.size(), false);
        std::vector<bool> armor_matched(armors.size(), false);
        for (std::size_t i = 0; i < trackers.size(); ++i) {
            if (trackers[i]->getState() == auto_aim::TrackerState::LOST) continue;
            int best_idx = -1;
            double best_dist = track_cfg.max_match_distance;
            for (std::size_t j = 0; j < armors.size(); ++j) {
                if (armor_matched[j] || !armors[j].solve_result) continue;
                double dist = distance3D(predicted_armors[i].tvec, armors[j].tvec);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            if (best_idx != -1) {
                trackers[i]->update(armors[best_idx], timestamp);
                tracker_matched[i] = true;
                armor_matched[best_idx] = true;
            } else {
                trackers[i]->lostUpdate();
            }
        }

        for (std::size_t j = 0; j < armors.size(); ++j) {
            if (!armor_matched[j] && armors[j].solve_result) {
                auto new_tracker = std::make_unique<auto_aim::Tracker>(track_cfg);
                new_tracker->init(armors[j], timestamp);
                trackers.push_back(std::move(new_tracker));
            }
        }

        trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
            [](const std::unique_ptr<auto_aim::Tracker>& t) {
                return t->getState() == auto_aim::TrackerState::LOST;
            }), trackers.end());

        // 选择最佳目标（距离最近）
        int best_tracker_idx = -1;
        double best_distance = 1e9;
        std::cout << "Trackers count: " << trackers.size() << std::endl;
        for (std::size_t i = 0; i < trackers.size(); ++i) {
            auto state = trackers[i]->getState();
            if (state == auto_aim::TrackerState::TRACKING || state == auto_aim::TrackerState::TEMP_LOST) {
                double dist = cv::norm(trackers[i]->getLatestArmor().tvec);
                if (dist < best_distance) {
                    best_distance = dist;
                    best_tracker_idx = i;
                }
            }
        }

        // ------------------- 可视化 -------------------
        // 灯条
        for (const auto& light : lights) {
            cv::circle(img, light.top, 1.5, cv::Scalar(0, 255, 255), 1);
            cv::circle(img, light.bottom, 1.5, cv::Scalar(0, 255, 255), 1);
            cv::line(img, light.top, light.bottom, cv::Scalar(0, 0, 255), 1);
        }
        // armor
        for (const auto& armor : armors){
            cv::line(img,armor.left.top,armor.right.bottom,cv::Scalar(0,255,2),1);
            cv::line(img,armor.left.bottom,armor.right.top,cv::Scalar(0,255,2),1);
        }
        // 所有跟踪器
        for (const auto& t : trackers) {
            if (t->getState() == auto_aim::TrackerState::LOST) continue;
            const auto& armor = t->getLatestArmor();
            std::vector<cv::Point2f> corners = {
                armor.left.top, armor.right.top,
                armor.right.bottom, armor.left.bottom
            };
            for (std::size_t i = 0; i < corners.size(); ++i)
                cv::line(img, corners[i], corners[(i+1)%4], cv::Scalar(0, 255, 0), 1);
            double dist = cv::norm(armor.tvec);
            cv::putText(img, std::to_string(dist) + "m", armor.left.top + cv::Point2f(-10, -10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        // 当前选中目标（红色边框）
        if (best_tracker_idx != -1) {
            const auto& armor = trackers[best_tracker_idx]->getLatestArmor();
            std::vector<cv::Point2f> corners = {
                armor.left.top, armor.right.top,
                armor.right.bottom, armor.left.bottom
            };
            for (std::size_t i = 0; i < corners.size(); ++i)
                cv::line(img, corners[i], corners[(i+1)%4], cv::Scalar(0, 0, 255), 1);
        }

        cv::resize(binary, binary, cv::Size(400, 400));
        cv::resize(img, img, cv::Size(400, 400));
        cv::imshow("Detection & Tracking", img);
        cv::imshow("Dilated", dilated);
        key = cv::waitKey(20);
        if (key == 27) break;
    }

    return 0;
}
