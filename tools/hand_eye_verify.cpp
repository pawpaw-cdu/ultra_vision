// Lightweight hand-eye projection check, inspired by sp_vision_25's
// tests/handeye_test.cpp. It projects a known board pose through the calibrated
// camera-to-gimbal transform; it does not solve calibration.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "tools/calibration/board.hpp"
#include "tools/calibration/pose.hpp"

namespace
{
    bool parseDoubles(const std::string& text, std::vector<double>& values)
    {
        values.clear();
        std::stringstream stream(text);
        std::string token;
        while (std::getline(stream, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (const std::exception&) {
                return false;
            }
        }
        return true;
    }

    cv::Mat matrixFromYaml(const YAML::Node& node)
    {
        if (node.IsMap()) {
            const int rows = node["rows"].as<int>();
            const int cols = node["cols"].as<int>();
            const auto values = node["data"].as<std::vector<double>>();
            if (rows <= 0 || cols <= 0 ||
                values.size() != static_cast<std::size_t>(rows * cols)) {
                return {};
            }
            cv::Mat matrix(rows, cols, CV_64F);
            for (int row = 0; row < rows; ++row) {
                for (int column = 0; column < cols; ++column) {
                    matrix.at<double>(row, column) =
                        values[static_cast<std::size_t>(row * cols + column)];
                }
            }
            return matrix;
        }
        const auto values = node.as<std::vector<double>>();
        return cv::Mat(values, true);
    }

} // namespace

int main(int argc, char** argv)
{
    std::string config_dir = "configs";
    std::string extrinsic = "hand_eye.yaml";
    std::string image_path;
    std::string output_path;
    std::string position_text = "1.0,0,0";
    std::string rpy_text = "0,0,0";
    double yaw_deg = 0.0;
    double pitch_deg = 0.0;
    bool display = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config-dir" && i + 1 < argc) config_dir = argv[++i];
        else if (arg == "--extrinsic" && i + 1 < argc) extrinsic = argv[++i];
        else if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--yaw" && i + 1 < argc) yaw_deg = std::atof(argv[++i]);
        else if (arg == "--pitch" && i + 1 < argc) pitch_deg = std::atof(argv[++i]);
        else if (arg == "--board-position" && i + 1 < argc) position_text = argv[++i];
        else if (arg == "--board-rpy" && i + 1 < argc) rpy_text = argv[++i];
        else if (arg == "--display") display = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: hand_eye_verify --image frame.jpg [options]\n"
                   "  --config-dir configs\n"
                   "  --extrinsic hand_eye.yaml\n"
                   "  --yaw <deg> --pitch <deg>\n"
                   "  --board-position x,y,z       board centre in base frame (m)\n"
                   "  --board-rpy roll,pitch,yaw   board orientation (deg)\n"
                   "  --out overlay.png --display\n";
            return 0;
        }
    }
    if (image_path.empty()) {
        std::cerr << "hand_eye_verify: --image is required\n";
        return 2;
    }
    const cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "hand_eye_verify: cannot read " << image_path << "\n";
        return 2;
    }

    std::vector<double> board_position;
    std::vector<double> board_rpy;
    if (!parseDoubles(position_text, board_position) || board_position.size() != 3 ||
        !parseDoubles(rpy_text, board_rpy) || board_rpy.size() != 3) {
        std::cerr << "hand_eye_verify: board position needs x,y,z and rpy needs roll,pitch,yaw\n";
        return 2;
    }

    const YAML::Node camera_file = YAML::LoadFile(config_dir + "/camera.yaml");
    const std::string camera_name =
        camera_file["camera"]["name"].as<std::string>("hikcamera");
    const cv::Mat camera_matrix =
        matrixFromYaml(camera_file[camera_name]["camera_matrix"]);
    const cv::Mat dist_coeffs =
        matrixFromYaml(camera_file[camera_name]["dist_coeffs"]);
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        std::cerr << "hand_eye_verify: camera_matrix must be 3x3\n";
        return 2;
    }

    const YAML::Node ext = YAML::LoadFile(extrinsic);
    const YAML::Node extrinsic_node =
        ext["R_camera2gimbal"] ? ext : ext[camera_name];
    if (!extrinsic_node || !extrinsic_node["R_camera2gimbal"] ||
        !extrinsic_node["t_camera2gimbal"]) {
        std::cerr << "hand_eye_verify: extrinsic YAML has no R_camera2gimbal/"
                     "t_camera2gimbal\n";
        return 2;
    }
    const auto rotation_values =
        extrinsic_node["R_camera2gimbal"].as<std::vector<double>>();
    const auto translation_values =
        extrinsic_node["t_camera2gimbal"].as<std::vector<double>>();
    if (rotation_values.size() != 9 || translation_values.size() != 3) {
        std::cerr << "hand_eye_verify: extrinsic needs R(9) + t(3)\n";
        return 2;
    }
    cv::Matx33d R_camera2gimbal;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            R_camera2gimbal(row, col) = rotation_values[static_cast<std::size_t>(row * 3 + col)];
        }
    }
    const cv::Vec3d t_camera2gimbal(translation_values[0], translation_values[1],
                                    translation_values[2]);
    const cv::Matx33d R_gimbal2base =
        calibration::gimbalToBase(yaw_deg * CV_PI / 180.0, pitch_deg * CV_PI / 180.0);
    const double roll = board_rpy[0] * CV_PI / 180.0;
    const double pitch = board_rpy[1] * CV_PI / 180.0;
    const double yaw = board_rpy[2] * CV_PI / 180.0;
    cv::Matx33d Rx, Ry, Rz;
    cv::Rodrigues(cv::Vec3d(roll, 0.0, 0.0), Rx);
    cv::Rodrigues(cv::Vec3d(0.0, pitch, 0.0), Ry);
    cv::Rodrigues(cv::Vec3d(0.0, 0.0, yaw), Rz);
    const cv::Matx33d R_board2base = Rz * Ry * Rx;
    const cv::Vec3d board_center(board_position[0], board_position[1], board_position[2]);

    const auto object_points =
        calibration::boardObjectPoints(calibration::loadBoardConfig(config_dir));
    std::vector<cv::Point3f> board_in_base;
    board_in_base.reserve(object_points.size());
    for (const cv::Point3f& point : object_points) {
        const cv::Vec3d p(point.x, point.y, point.z);
        const cv::Vec3d in_base = board_center + R_board2base * p;
        board_in_base.emplace_back(static_cast<float>(in_base[0]),
                                   static_cast<float>(in_base[1]),
                                   static_cast<float>(in_base[2]));
    }
    std::vector<cv::Point3f> board_in_camera;
    board_in_camera.reserve(board_in_base.size());
    for (const cv::Point3f& point : board_in_base) {
        const cv::Vec3d in_gimbal = R_gimbal2base.t() * cv::Vec3d(point.x, point.y, point.z);
        const cv::Vec3d in_camera = R_camera2gimbal.t() * (in_gimbal - t_camera2gimbal);
        board_in_camera.emplace_back(static_cast<float>(in_camera[0]),
                                     static_cast<float>(in_camera[1]),
                                     static_cast<float>(in_camera[2]));
    }
    std::vector<cv::Point2f> projected;
    cv::projectPoints(board_in_camera, cv::Vec3d::zeros(), cv::Vec3d::zeros(),
                      camera_matrix, dist_coeffs, projected);
    if (projected.empty() || !std::isfinite(projected.front().x) ||
        !std::isfinite(projected.front().y)) {
        std::cerr << "hand_eye_verify: projection produced no finite points\n";
        return 1;
    }

    cv::Mat overlay = image.clone();
    int visible = 0;
    for (const cv::Point2f& point : projected) {
        if (point.x < 0 || point.y < 0 || point.x >= overlay.cols || point.y >= overlay.rows) {
            continue;
        }
        cv::circle(overlay, point, 4, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
        ++visible;
    }
    cv::putText(overlay, "projected=" + std::to_string(visible) +
                             " yaw=" + std::to_string(static_cast<int>(yaw_deg)) +
                             " pitch=" + std::to_string(static_cast<int>(pitch_deg)),
                cv::Point(16, 32), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);
    if (!output_path.empty()) {
        cv::imwrite(output_path, overlay);
        std::cout << "wrote " << output_path << "\n";
    }
    std::cout << "visible points: " << visible << "/" << projected.size() << "\n";
    if (visible == 0) {
        std::cout << "first projected point: " << projected.front()
                  << " image=" << image.cols << "x" << image.rows << "\n";
    }
    if (display) {
        cv::imshow("hand_eye_verify", overlay);
        cv::waitKey(0);
    }
    return visible > 0 ? 0 : 1;
}
