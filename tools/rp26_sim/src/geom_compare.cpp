// 离线几何对照：在同一批录制帧上并排量"深大轮廓几何"和"我们的关键点几何"。
//
// 输入：RP26_DUMP_DIR 里由 rp26_sim 落盘的 raw_f*.png（文件名带仿真时间戳）
// 输出：CSV，每帧一行：
//   ours_*   = 我们链路现在的几何（关键点：k2 = 圆心，四角均值 = 靶心）
//   theirs_* = 他们的几何（颜色差轮廓 -> fitEllipse(R标).center = 圆心，装甲板轮廓 = 靶心）
//   还有他们的可用性判据（装甲板/灯臂/R标是否 usable）
//
// 用法：
//   rp26_geom_compare --dir /tmp/frames --csv /tmp/geom.csv [--out-limit N]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "RuneObservationRefiner.hpp"
#include "common/power_rune_function.hpp"
#include "json.hpp"
#include "power_rune_interface.hpp"
#include "transform_tools/transform_tools.h"

#include "rp26_detector.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
double deg(double rad) { return rad * 180.0 / kPi; }

struct FrameEntry
{
    std::filesystem::path path;
    std::uint64_t timestamp_us = 0;
};

std::vector<FrameEntry> collectFrames(const std::string &dir, const std::string &pattern)
{
    std::vector<FrameEntry> frames;
    for (const auto &entry : std::filesystem::directory_iterator(dir))
    {
        const std::string name = entry.path().filename().string();
        if (name.rfind(pattern, 0) != 0)
            continue;
        FrameEntry frame;
        frame.path = entry.path();
        // raw_f00012_345_1790168802123456.png -> 末段是仿真曝光时间戳（微秒）
        const std::size_t last = name.rfind('_');
        const std::size_t dot = name.rfind('.');
        if (last != std::string::npos && dot != std::string::npos && dot > last)
            frame.timestamp_us = std::strtoull(name.substr(last + 1, dot - last - 1).c_str(),
                                               nullptr, 10);
        if (frame.timestamp_us == 0)
        {
            // 我们链路 dump 的 frame_%06d.jpg 没有时间戳：按 30 fps 合成一个，
            // 只用于排序与相位斜坡分析（分辨率 1/30 s 足够）。
            const std::size_t first = name.find_first_of("0123456789");
            const std::size_t last_digit = name.find_first_not_of("0123456789", first);
            const unsigned long long index =
                first == std::string::npos
                    ? frames.size()
                    : std::strtoull(name.substr(first, last_digit - first).c_str(), nullptr, 10);
            frame.timestamp_us = 1000000ULL + index * 33333ULL;
        }
        frames.push_back(frame);
    }
    std::sort(frames.begin(), frames.end(),
              [](const FrameEntry &a, const FrameEntry &b) { return a.timestamp_us < b.timestamp_us; });
    return frames;
}

/// @brief 他们的几何：把 RuneObservation 交给他们的 refiner，取约束后的轮廓。
struct TheirGeometry
{
    bool has_center_r = false;
    bool has_armor = false;
    bool armor_usable = false;
    bool light_arm_usable = false;
    cv::Point2f r_center{0.0F, 0.0F};
    cv::Point2f armor_center{0.0F, 0.0F};
    double armor_area = 0.0;
};

TheirGeometry refineWithTheirs(RuneObservationRefiner &refiner, const cv::Mat &image,
                               const std::vector<power_rune::RuneInput::NNRuneInfo> &infos,
                               const transform_tools::TFTree &tree,
                               const timetool::Timestamp &stamp)
{
    TheirGeometry geometry;

    RuneObservation observation;
    observation.is_big_rune = false;
    observation.ori_img = image;
    observation.timestamp = power_rune_function::timestamp_to_foxglove_time(stamp);
    observation.tf_tree = tree;
    // 与他们 convert2rune_observation 一致：按五点 minAreaRect×1.4 裁 ROI，
    // 因为他们的 refiner 是在这个 ROI（view）里做颜色差二值化的。
    for (const auto &info : infos)
    {
        std::vector<cv::Point2i> points{info.top, info.left, info.point_R, info.right,
                                        info.bottom};
        cv::RotatedRect rotated_rect = cv::minAreaRect(points);
        rotated_rect.size = rotated_rect.size * static_cast<float>(
                                                  J_POWER_RUNE.config_["detect"]["extand_rotated_rect"]);
        cv::Rect view_rect = rotated_rect.boundingRect();
        view_rect &= cv::Rect(0, 0, image.cols, image.rows);
        if (view_rect.empty())
            continue;

        RuneInfo rune_info;
        rune_info.top = info.top;
        rune_info.left = info.left;
        rune_info.right = info.right;
        rune_info.bottom = info.bottom;
        rune_info.point_R = info.point_R;
        rune_info.class_id = info.class_id;
        rune_info.rotated_rect = rotated_rect;
        rune_info.view_rect = view_rect;
        rune_info.view = image(view_rect).clone();
        rune_info.color = RuneInfo::RED;
        observation.rune_infos.push_back(rune_info);
    }

    const RefinedRuneObservation refined = refiner.refine(observation);
    if (refined.rune_blade_2D.empty())
        return geometry;

    const SingleRuneBlade2D &blade = refined.rune_blade_2D.front();
    const auto &contours = blade.constrained_contours;
    if (contours.center_R_opt.has_value() && contours.center_R_opt->size() >= 5)
    {
        const cv::RotatedRect ellipse = cv::fitEllipse(contours.center_R_opt.value());
        geometry.r_center = ellipse.center;
        geometry.has_center_r = true;
    }
    if (contours.armor_module_opt.has_value() && contours.armor_module_opt->size() >= 5)
    {
        geometry.has_armor = true;
        geometry.armor_usable = blade.is_armor_module_usable;
        geometry.light_arm_usable = blade.is_light_arm_usable;
        geometry.armor_area = std::abs(cv::contourArea(contours.armor_module_opt.value()));
        const cv::Moments moments = cv::moments(contours.armor_module_opt.value());
        if (std::abs(moments.m00) > 1e-6)
            geometry.armor_center = cv::Point2f(static_cast<float>(moments.m10 / moments.m00),
                                                static_cast<float>(moments.m01 / moments.m00));
    }
    return geometry;
}
} // namespace

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    std::string dir;
    std::string csv_path;
    std::string pattern = "raw_";
    int limit = 0;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        const auto next = [&]() -> std::string { return argv[++index]; };
        if (arg == "--dir")
            dir = next();
        else if (arg == "--csv")
            csv_path = next();
        else if (arg == "--pattern")
            pattern = next();
        else if (arg == "--limit")
            limit = std::atoi(next().c_str());
        else
        {
            std::cerr << "usage: rp26_geom_compare --dir <frame dir> --csv <out.csv> [--limit N]"
                      << std::endl;
            return 2;
        }
    }
    if (dir.empty() || csv_path.empty())
    {
        std::cerr << "usage: rp26_geom_compare --dir <frame dir> --csv <out.csv> [--limit N]"
                  << std::endl;
        return 2;
    }

    const std::vector<FrameEntry> frames = collectFrames(dir, pattern);
    std::cout << "[geom_compare] frames=" << frames.size() << " from " << dir << std::endl;
    if (frames.empty())
        return 1;

    rp26_sim::Rp26RuneDetector detector;
    RuneObservationRefiner refiner;

    transform_tools::TFTree tree;
    tree.add_TF(transform_tools::TF(), car_frame, ecs_world_frame);
    tree[ecs_world_frame].set_rotation(Eigen::Matrix3d::Identity());
    tree.add_TF(transform_tools::TF(), ecs_world_frame, gimbal_frame);
    tree.add_TF(transform_tools::TF(), gimbal_frame, camera_frame);
    {
        Eigen::Matrix3d gimbal_from_camera;
        gimbal_from_camera << 0.0, 0.0, -1.0,
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0;
        tree[camera_frame].set_rotation(gimbal_from_camera);
    }
    tree.add_TF(transform_tools::TF(), car_frame, unbiased_camera_frame);

    std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
    csv << "idx,det_idx,det_n,is_best,t_sim_us,class_id,"
           "ours_k2_x,ours_k2_y,ours_plate_x,ours_plate_y,ours_plate_r,ours_phase_deg,"
           "theirs_centerR_x,theirs_centerR_y,theirs_armor_x,theirs_armor_y,theirs_armor_area,"
           "theirs_has_centerR,theirs_has_armor,theirs_armor_usable,theirs_lightarm_usable,"
           "dR_px,dPhase_deg\n";

    int written = 0;
    for (const FrameEntry &entry : frames)
    {
        if (limit > 0 && written >= limit)
            break;
        const cv::Mat image = cv::imread(entry.path.string(), cv::IMREAD_COLOR);
        if (image.empty())
            continue;

        std::vector<rp26_sim::Rp26Detection> detections;
        try
        {
            detections = detector.infer(image);
        }
        catch (const std::exception &error)
        {
            std::cerr << "[geom_compare] infer failed: " << error.what() << std::endl;
            continue;
        }
        if (detections.empty())
            continue;

        // 我们链路只认 class 0（未击打）的那片；这里也取 class 0 优先。
        std::size_t best_index = 0;
        for (std::size_t index = 0; index < detections.size(); ++index)
            if (detections[index].model_class_id == 0)
            {
                best_index = index;
                break;
            }

        // 每帧按"每一条候选"各写一行：大符每轮两片同时点亮，需要看两条候选的
        // 靶心/R标点分布，才能判断"跳动"是选靶切换还是网络关键点本身在跳。
        for (std::size_t index = 0; index < detections.size(); ++index)
        {
            const bool is_best = index == best_index;
            const rp26_sim::Rp26Detection &detection = detections[index];
            const cv::Point2f k2 = detection.keypoints[2];
            const cv::Point2f plate =
                0.25F * (detection.keypoints[0] + detection.keypoints[1] +
                         detection.keypoints[3] + detection.keypoints[4]);
            double plate_radius = 0.0;
            for (int point_index : {0, 1, 3, 4})
                plate_radius += cv::norm(detection.keypoints[point_index] - plate);
            plate_radius /= 4.0;
            const double ours_phase = deg(std::atan2(plate.y - k2.y, plate.x - k2.x));

            TheirGeometry theirs;
            if (is_best)
            {
                const auto infos = rp26_sim::Rp26RuneDetector::toRuneInfos(
                    detections, rp26_sim::Rp26RuneMode::Small);
                const timetool::Timestamp stamp{
                    std::chrono::duration_cast<std::chrono::system_clock::duration>(
                        std::chrono::microseconds(entry.timestamp_us))};
                theirs = refineWithTheirs(refiner, image, infos, tree, stamp);
            }

            double d_r = -1.0;
            double d_phase = 0.0;
            if (theirs.has_center_r)
            {
                d_r = cv::norm(cv::Point2f(theirs.r_center.x - k2.x, theirs.r_center.y - k2.y));
                if (theirs.has_armor)
                {
                    const double theirs_phase =
                        deg(std::atan2(theirs.armor_center.y - theirs.r_center.y,
                                       theirs.armor_center.x - theirs.r_center.x));
                    d_phase = theirs_phase - ours_phase;
                    while (d_phase > 180.0)
                        d_phase -= 360.0;
                    while (d_phase < -180.0)
                        d_phase += 360.0;
                }
            }

            csv << written << ',' << index << ',' << detections.size() << ','
                << (is_best ? 1 : 0) << ',' << entry.timestamp_us << ','
                << detection.model_class_id << ',' << k2.x << ',' << k2.y << ',' << plate.x
                << ',' << plate.y << ',' << plate_radius << ',' << ours_phase << ','
                << theirs.r_center.x << ',' << theirs.r_center.y << ',' << theirs.armor_center.x
                << ',' << theirs.armor_center.y << ',' << theirs.armor_area << ','
                << (theirs.has_center_r ? 1 : 0) << ',' << (theirs.has_armor ? 1 : 0) << ','
                << (theirs.armor_usable ? 1 : 0) << ',' << (theirs.light_arm_usable ? 1 : 0)
                << ',' << d_r << ',' << d_phase << '\n';
        }
        ++written;
    }

    std::cout << "[geom_compare] wrote " << written << " rows -> " << csv_path << std::endl;
    return 0;
}
