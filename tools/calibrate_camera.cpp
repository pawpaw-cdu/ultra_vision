// 相机内参标定（棋盘格照片 → camera_matrix / dist_coeffs），对齐 sp_vision 的
// calibration/calibrate_camera.cpp：给一个装满 {1.jpg,2.jpg,...} 的文件夹即可。
//
//   tools/calibrate_camera <图片文件夹> [--cols 11] [--rows 8] [--size 15]
//                          [--out camera_intrinsics.yaml] [--selftest]
//   tools/calibrate_camera <图片文件夹> --live      # 直接开相机，空格存一张（含标定板检测提示），
//                                                  # q 退出后自动跑标定（省掉"先拍照再喂文件夹"）
//
// 为什么需要：`configs/camera.yaml` 里海康那组内参是"按 1280x960 年代换算的估计值"，
// 重投影误差会直接变成距离误差（1 px ≈ 0.5 m @7.5 m）。标一次写回去最省事。

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#if defined(ULTRA_VISION_USE_HIK_CAMERA)
#include "io/camera/HikCamera.hpp"
using CameraType = rm_ultra::HikCamera;
#elif defined(ULTRA_VISION_USE_GALAXY_CAMERA)
#include "io/camera/GalaxyCamera.hpp"
using CameraType = rm_ultra::GalaxyCamera;
#endif

namespace
{
    std::vector<cv::Point3f> boardPoints(const cv::Size& pattern, double square_mm)
    {
        std::vector<cv::Point3f> points;
        points.reserve(static_cast<std::size_t>(pattern.area()));
        for (int row = 0; row < pattern.height; ++row) {
            for (int col = 0; col < pattern.width; ++col) {
                points.emplace_back(static_cast<float>(col * square_mm),
                                    static_cast<float>(row * square_mm), 0.0f);
            }
        }
        return points;
    }

    /// @brief 用合成棋盘格做自检：已知内参 → 造图 → 标定 → 看能否复原。
    int selftest()
    {
        const cv::Size pattern(9, 6);
        const double square_mm = 20.0;
        const cv::Matx33d truth(1200.0, 0.0, 720.0, 0.0, 1200.0, 540.0, 0.0, 0.0, 1.0);
        const std::vector<cv::Point3f> object = boardPoints(pattern, square_mm);
        std::vector<std::vector<cv::Point3f>> object_points;
        std::vector<std::vector<cv::Point2f>> image_points;
        cv::Mat distortion;   // 合成数据不带畸变
        for (int view = 0; view < 12; ++view) {
            const double angle = view * 0.37;
            const cv::Vec3d rvec(std::sin(angle) * 0.4, std::cos(angle) * 0.3, angle * 0.15);
            const cv::Vec3d tvec(std::sin(angle) * 120.0, std::cos(angle) * 90.0, 700.0 + 60.0 * view);
            std::vector<cv::Point2f> projected;
            cv::projectPoints(object, rvec, tvec, truth, distortion, projected);
            object_points.push_back(object);
            image_points.push_back(projected);
        }
        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat dist_coeffs;
        const double rms = cv::calibrateCamera(object_points, image_points, cv::Size(1440, 1080),
                                              camera_matrix, dist_coeffs, cv::noArray(),
                                              cv::noArray());
        const double fx_error = std::abs(camera_matrix.at<double>(0, 0) - truth(0, 0));
        const double fy_error = std::abs(camera_matrix.at<double>(1, 1) - truth(1, 1));
        std::printf("calibrate_camera --selftest: rms=%.4f px, fx 误差 %.3f, fy 误差 %.3f\n",
                    rms, fx_error, fy_error);
        if (rms > 1.0 || fx_error > 5.0 || fy_error > 5.0) {
            std::printf("FAILED: 合成数据都没标回来（rms/fx/fy 超限）\n");
            return 1;
        }
        std::printf("通过\n");
        return 0;
    }
    /// @brief 读一个文件夹的棋盘格照片 → 内参 → 打印并写 yaml（--live 采集完也走这里）。
    int calibrateFolder(const std::string& folder, const cv::Size& pattern, double square_mm,
                        const std::string& output, const std::string& kind = "chessboard")
    {
    const std::vector<cv::Point3f> object = boardPoints(pattern, square_mm);
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    cv::Size image_size;

    std::vector<cv::String> files;
    cv::glob(folder + "/*.jpg", files, false);
    std::sort(files.begin(), files.end());
    for (const auto& path : files) {
        const cv::Mat image = cv::imread(path);
        if (image.empty()) continue;
        std::vector<cv::Point2f> corners;
        bool found = false;
        if (kind == "circles") {
            // 圆点阵：findCirclesGrid 给的就是亚像素质心，**不能**再跑 cornerSubPix
            // （它会找"角点"，把圆心拖到圆盘边缘/白底上，实测能拖走 3.9 px、最大 11.7 px）。
            found = cv::findCirclesGrid(image, pattern, corners, cv::CALIB_CB_SYMMETRIC_GRID);
        } else {
            // 先 SB（对光照/模糊更稳），不再用经典检测的 FAST_CHECK 提前放弃——实测同一批
            // 30 张里经典+FAST_CHECK 只认出 24 张，SB 全中。经典检测留作兜底。
            found = cv::findChessboardCornersSB(image, pattern, corners,
                                                cv::CALIB_CB_NORMALIZE_IMAGE);
            if (!found) {
                found = cv::findChessboardCorners(
                    image, pattern, corners,
                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            }
        }
        if (!found) continue;
        if (kind != "circles") {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                                              0.01));
        }
        object_points.push_back(object);
        image_points.push_back(corners);
        image_size = image.size();
        std::cout << "[calib] 采用 " << path << "（累计 " << image_points.size() << " 张）"
                  << std::endl;
    }
    if (image_points.size() < 8) {
        std::cerr << "有效棋盘格少于 8 张（" << image_points.size() << "），标不准；多拍几张不同角度"
                  << std::endl;
        return 1;
    }

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix,
                                     dist_coeffs, rvecs, tvecs);
    // 逐视角体检：用解出来的内参对每张图做一次 PnP，平均重投影误差特别大的那张多半是角点
    // 顺序错或糊图。剔掉离群的再解一次（只踢明显离群，不踢正常噪声）。
    auto perViewErrors = [&](const cv::Mat& K, const cv::Mat& D) {
        std::vector<double> errors;
        errors.reserve(image_points.size());
        for (size_t i = 0; i < image_points.size(); ++i) {
            cv::Mat rvec, tvec;
            if (!cv::solvePnP(object_points[i], image_points[i], K, D, rvec, tvec)) {
                errors.push_back(1e9);
                continue;
            }
            std::vector<cv::Point2f> projected;
            cv::projectPoints(object_points[i], rvec, tvec, K, D, projected);
            errors.push_back(cv::norm(projected, image_points[i], cv::NORM_L2) /
                             std::sqrt(static_cast<double>(projected.size())));
        }
        return errors;
    };
    std::vector<double> errors = perViewErrors(camera_matrix, dist_coeffs);
    std::vector<double> sorted = errors;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    const double limit = std::max(0.5, 3.0 * median);
    std::vector<std::vector<cv::Point3f>> keep_object;
    std::vector<std::vector<cv::Point2f>> keep_image;
    for (size_t i = 0; i < errors.size(); ++i) {
        if (errors[i] > limit) {
            std::cout << "[calib] 剔除第 " << (i + 1) << " 张（单帧重投影 " << errors[i]
                      << " px > " << limit << " px）" << std::endl;
            continue;
        }
        keep_object.push_back(object_points[i]);
        keep_image.push_back(image_points[i]);
    }
    if (keep_image.size() >= 8 && keep_image.size() < image_points.size()) {
        object_points.swap(keep_object);
        image_points.swap(keep_image);
        rms = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs,
                                  rvecs, tvecs);
        errors = perViewErrors(camera_matrix, dist_coeffs);
        std::cout << "[calib] 剔除离群视角后剩 " << image_points.size() << " 张，rms = " << rms
                  << " px" << std::endl;
    }
    double worst = 0.0;
    for (const double e : errors) worst = std::max(worst, e);
    std::cout << "[calib] 重投影 rms = " << rms << " px（<0.3 算好，>1 要查标定板/对焦）"
              << std::endl;
    std::cout << "[calib] 单帧重投影 中位 " << median << " px / 最差 " << worst << " px（"
              << image_points.size() << " 张）" << std::endl;
    std::cout << "[calib] camera_matrix = [" << camera_matrix.at<double>(0, 0) << ", 0, "
              << camera_matrix.at<double>(0, 2) << ", 0, " << camera_matrix.at<double>(1, 1)
              << ", " << camera_matrix.at<double>(1, 2) << ", 0, 0, 1]" << std::endl;

    std::ofstream out(output);
    out << "# 由 tools/calibrate_camera 生成（" << image_size.width << "x" << image_size.height
        << "，rms " << rms << " px）—— 粘到 configs/camera.yaml 的 hikcamera/galaxy 段\n";
    out << "camera_matrix: [" << camera_matrix.at<double>(0, 0) << ",0," << camera_matrix.at<double>(0, 2)
        << ",0," << camera_matrix.at<double>(1, 1) << "," << camera_matrix.at<double>(1, 2)
        << ",0,0,1]\n";
    out << "dist_coeffs: [";
    for (int i = 0; i < dist_coeffs.total(); ++i) {
        out << (i ? "," : "") << dist_coeffs.at<double>(i);
    }
    out << "]\n";
    std::cout << "[calib] 已写 " << output << std::endl;
    return rms < 1.0 ? 0 : 1;
    }
} // namespace

#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
/// @brief 现场采集：开相机，空格存图（只在找到棋盘格时才存，避免混入废图），q 退出。
/// @param auto_quit_seconds >0 时到点自动结束采集（等价于按 q），方便无人值守/无键盘时跑通整条链。
int captureLive(const std::string& folder, const cv::Size& pattern, double square_mm,
                double auto_quit_seconds = 0.0, const std::string& kind = "chessboard")
{
    const std::string config_dir = std::getenv("ULTRA_VISION_CONFIG_DIR")
        ? std::getenv("ULTRA_VISION_CONFIG_DIR") : "configs";
    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const auto camera_node = camera_file["camera"];
    CameraType camera;
    const int device_index = camera_node["device_index"].as<int>(1);
    if (!camera.init("", device_index)) {
        std::cerr << "[calib] 打不开相机" << std::endl;
        return -1;   // -1 = 相机没打开（和"张数不够"区分开）
    }
    if (const YAML::Node hik = camera_file["hikcamera"]) {
        camera.setAutoExposure(hik["auto_exposure"].as<bool>(false));
        camera.setExposureTime(hik["exposure_ms"].as<double>(6.0) * 1000.0);
        camera.setGain(hik["gain"].as<double>(12.0));
    }
    std::system(("mkdir -p " + folder).c_str());
    cv::Mat frame;
    // 续编已有编号：重复跑 --live 时不会覆盖上一次拍的图（否则第 N 张永远从 1 开始，把旧图冲掉）。
    int saved = 0;
    {
        std::vector<cv::String> existing;
        cv::glob(folder + "/*.jpg", existing, false);
        for (const auto& path : existing) {
            const std::string stem = path.substr(path.find_last_of('/') + 1);
            const std::string digits = stem.substr(0, stem.find('.'));
            if (digits.empty() ||
                digits.find_first_not_of("0123456789") != std::string::npos) {
                continue;
            }
            saved = std::max(saved, std::atoi(digits.c_str()));
        }
        if (saved > 0) {
            std::cout << "[calib] " << folder << " 里已有 " << saved << " 张，本次从 "
                      << saved + 1 << " 继续编号" << std::endl;
        }
    }
    std::cout << "[calib] 空格=存一张（只在该帧找到棋盘格时才存），q/ESC=退出。"
              << "要在不同角度/位置拍 15~30 张，覆盖画面四角。" << std::endl;

    // 性能：**检测只在 640 宽的预览图上做、且隔帧**。经典 findChessboardCorners 在
    // 1440x1080 上要几百 ms（实测 353 ms/帧），每帧跑会把帧率压到个位数。
    // 存图那一刻才在**全分辨率**上复检一次（一次按键一次，代价可接受），
    // 保证存下来的是能用的高清图。
    constexpr int kPreviewWidth = 640;
    static bool board_found = false;
    static std::vector<cv::Point2f> board_corners;
    cv::Mat preview;
    cv::Mat preview_gray;
    uint64_t frame_index = 0;
    int preview_frames = 0;
    int added = 0;
    const auto live_started = std::chrono::steady_clock::now();
    double detect_ms = 0.0;
    auto fps_timer = std::chrono::steady_clock::now();
    while (true) {
        if (auto_quit_seconds > 0.0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - live_started).count() >=
                auto_quit_seconds) {
            break;
        }
        if (!camera.getImage(frame, 500) || frame.empty()) continue;
        ++frame_index;

        const bool scaled = frame.cols > kPreviewWidth;
        const double scale = scaled ? static_cast<double>(kPreviewWidth) / frame.cols : 1.0;
        if (scaled) {
            cv::resize(frame, preview, cv::Size(), scale, scale, cv::INTER_AREA);
        } else {
            preview = frame;
        }

        const auto before_detect = std::chrono::steady_clock::now();
        const bool run_detect = (frame_index % 2 == 1);
        if (run_detect) {
            board_corners.clear();
            // 预览阶段只用 SB（快，~20 ms @640）；不在预览上跑经典检测
            board_found = (kind == "circles")
                ? cv::findCirclesGrid(preview, pattern, board_corners, cv::CALIB_CB_SYMMETRIC_GRID)
                : cv::findChessboardCornersSB(preview, pattern, board_corners,
                                              cv::CALIB_CB_NORMALIZE_IMAGE);
            detect_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - before_detect).count();
        }

        if (board_found) {
            cv::drawChessboardCorners(preview, pattern, board_corners, true);
        }
        cv::putText(preview,
                    cv::format("saved=%d  board=%s  detect=%.0fms  [space]=save [q]=quit", saved,
                               board_found ? "OK" : "--", detect_ms),
                    cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    board_found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::imshow("calibrate_camera", preview);
        const int key = cv::waitKey(1);

        if (++preview_frames % 30 == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - fps_timer).count();
            if (elapsed >= 1.0) {
                std::cout << "[calib] 预览 " << preview_frames / elapsed << " fps（detect "
                          << detect_ms << " ms）" << std::endl;
                preview_frames = 0;
                fps_timer = now;
            }
        }

        if (key == 27 || key == 'q' || key == 'Q') break;
        if (key != ' ') continue;
        if (!board_found) {
            std::cout << "[calib] 这一帧没找到棋盘格，不存（先让板子完整入画、别太斜）" << std::endl;
            continue;
        }
        // 存图前在**全分辨率**上复检（含经典检测兜底），避免存进糊图/半张板
        std::vector<cv::Point2f> full_corners;
        bool full_found = (kind == "circles")
            ? cv::findCirclesGrid(frame, pattern, full_corners, cv::CALIB_CB_SYMMETRIC_GRID)
            : cv::findChessboardCornersSB(frame, pattern, full_corners,
                                          cv::CALIB_CB_NORMALIZE_IMAGE);
        if (!full_found && kind != "circles") {
            full_found = cv::findChessboardCorners(
                frame, pattern, full_corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        }
        if (!full_found) {
            std::cout << "[calib] 全分辨率复检没通过，跳过这张" << std::endl;
            continue;
        }
        ++saved;
        ++added;
        const std::string path = folder + "/" + std::to_string(saved) + ".jpg";
        cv::imwrite(path, frame);
        std::cout << "[calib] 存第 " << saved << " 张（" << path << "，全分辨率 " << frame.cols << "x"
                  << frame.rows << "）" << std::endl;
    }
    std::vector<cv::String> total;
    cv::glob(folder + "/*.jpg", total, false);
    std::cout << "[calib] 采集结束：本次新存 " << added << " 张，文件夹内累计 " << total.size()
              << " 张" << std::endl;
    // 回传文件夹里的总张数：main 用它判断"够不够 8 张、要不要接着标定"。
    return static_cast<int>(total.size());
}
#endif

int main(int argc, char* argv[])
{
    std::string folder;
    int cols = 11;
    int rows = 8;
    double square_mm = 15.0;
    std::string output = "camera_intrinsics.yaml";
    bool live = false;
    std::string kind = "chessboard";   // chessboard | circles
    double live_seconds = 0.0;   // --live 到点自动结束（0 = 按 q 才结束）
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--live") live = true;
        else if (arg == "--pattern" && i + 1 < argc) kind = argv[++i];
        else if (arg == "--selftest") return selftest();
        else if (arg == "--live-seconds" && i + 1 < argc) live_seconds = std::atof(argv[++i]);
        else if (arg == "--cols" && i + 1 < argc) cols = std::atoi(argv[++i]);
        else if (arg == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (arg == "--size" && i + 1 < argc) square_mm = std::atof(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) output = argv[++i];
        else if (folder.empty()) folder = arg;
    }
    if (folder.empty()) {
        std::cerr << "用法: calibrate_camera <图片文件夹> [--cols 11] [--rows 8] [--size 15] "
                     "[--out camera_intrinsics.yaml] [--live] [--live-seconds 0] [--pattern chessboard|circles]"
                     " [--selftest]\n";
        return 2;
    }
    const cv::Size pattern_live(cols, rows);
    if (live) {
#if defined(ULTRA_VISION_USE_HIK_CAMERA) || defined(ULTRA_VISION_USE_GALAXY_CAMERA)
        const int saved = captureLive(folder, pattern_live, square_mm, live_seconds, kind);
        if (saved < 8) {
            std::cerr << "[calib] 只采集到 " << saved << " 张（至少 8 张，建议 15~30 张），"
                         "这次不标定" << std::endl;
            return 1;
        }
        std::cout << "[calib] 采集结束（" << saved << " 张），继续自动标定…" << std::endl;
        return calibrateFolder(folder, pattern_live, square_mm, output, kind);
#else
        std::cerr << "--live 需要带相机 SDK 构建" << std::endl;
        return 2;
#endif
    }

    return calibrateFolder(folder, pattern_live, square_mm, output, kind);
}
