#include "tools/calibration/board.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace calibration
{
    BoardConfig loadBoardConfig(const std::string& config_dir)
    {
        BoardConfig board;
        const std::string path = config_dir + "/calibration.yaml";
        std::ifstream probe(path);
        if (!probe) return board;
        const YAML::Node file = YAML::LoadFile(path);
        const YAML::Node node = file["board"] ? file["board"] : file;
        if (!node) return board;
        board.pattern = node["pattern"].as<std::string>(board.pattern);
        board.cols = node["cols"].as<int>(board.cols);
        board.rows = node["rows"].as<int>(board.rows);
        board.square_mm = node["square_mm"].as<double>(board.square_mm);
        board.grid = node["grid"].as<std::string>(board.grid);
        return board;
    }

    std::vector<cv::Point3f> boardObjectPoints(const BoardConfig& board,
                                               double unit_scale)
    {
        std::vector<cv::Point3f> points;
        points.reserve(static_cast<std::size_t>(board.cols * board.rows));
        const float step = static_cast<float>(board.square_mm * unit_scale);
        for (int row = 0; row < board.rows; ++row) {
            for (int col = 0; col < board.cols; ++col) {
                points.emplace_back(col * step, row * step, 0.0f);
            }
        }
        return points;
    }

    bool detectBoardCorners(const cv::Mat& image, const BoardConfig& board,
                            std::vector<cv::Point2f>& corners, bool refine_subpixel,
                            bool allow_classic_fallback)
    {
        if (image.empty()) return false;
        const cv::Size pattern(board.cols, board.rows);
        bool found = false;
        if (board.pattern == "circles") {
            const int flags = board.grid == "asymmetric" ? cv::CALIB_CB_ASYMMETRIC_GRID
                                                         : cv::CALIB_CB_SYMMETRIC_GRID;
            found = cv::findCirclesGrid(image, pattern, corners, flags);
        } else {
            found = cv::findChessboardCornersSB(
                image, pattern, corners, cv::CALIB_CB_NORMALIZE_IMAGE);
            if (!found && allow_classic_fallback) {
                found = cv::findChessboardCorners(
                    image, pattern, corners,
                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            }
        }
        if (!found) return false;
        // A circle grid already returns subpixel centres. cornerSubPix looks for
        // saddle corners and systematically drags circle centres toward edges.
        if (refine_subpixel && board.pattern != "circles") {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                                              0.01));
        }
        return true;
    }

    bool detectBoard(const cv::Mat& image, const BoardConfig& board,
                     const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                     BoardObservation& observation, bool allow_classic_fallback)
    {
        if (!detectBoardCorners(image, board, observation.corners, true,
                                allow_classic_fallback)) {
            return false;
        }
        const auto object_points = boardObjectPoints(board);
        auto reprojectionRms = [&](const std::vector<cv::Point2f>& points,
                                   const cv::Mat& r, const cv::Mat& t) {
            std::vector<cv::Point2f> projected;
            cv::projectPoints(object_points, r, t, camera_matrix, dist_coeffs, projected);
            double sum = 0.0;
            for (std::size_t i = 0; i < points.size(); ++i) {
                const double distance = cv::norm(points[i] - projected[i]);
                sum += distance * distance;
            }
            return std::sqrt(sum / static_cast<double>(points.size()));
        };

        std::vector<cv::Point2f> reordered(observation.corners.size());
        for (int row = 0; row < board.rows; ++row) {
            for (int col = 0; col < board.cols; ++col) {
                reordered[static_cast<std::size_t>(row) * board.cols + col] =
                    observation.corners[static_cast<std::size_t>(col) * board.rows + row];
            }
        }
        cv::Mat rvec_direct, tvec_direct, rvec_transposed, tvec_transposed;
        const bool ok_direct = cv::solvePnP(
            object_points, observation.corners, camera_matrix, dist_coeffs,
            rvec_direct, tvec_direct, false, cv::SOLVEPNP_ITERATIVE);
        const bool ok_transposed = cv::solvePnP(
            object_points, reordered, camera_matrix, dist_coeffs,
            rvec_transposed, tvec_transposed, false, cv::SOLVEPNP_ITERATIVE);
        if (!ok_direct && !ok_transposed) return false;
        const double rms_direct =
            ok_direct ? reprojectionRms(observation.corners, rvec_direct, tvec_direct)
                      : std::numeric_limits<double>::max();
        const double rms_transposed =
            ok_transposed ? reprojectionRms(reordered, rvec_transposed, tvec_transposed)
                          : std::numeric_limits<double>::max();
        if (std::getenv("ULTRA_VISION_HAND_EYE_DEBUG")) {
            std::printf("[debug] direct rms=%.4f t=(%.3f,%.3f,%.3f) | "
                        "transposed rms=%.4f t=(%.3f,%.3f,%.3f)\n",
                        rms_direct, tvec_direct.at<double>(0), tvec_direct.at<double>(1),
                        tvec_direct.at<double>(2), rms_transposed,
                        tvec_transposed.at<double>(0), tvec_transposed.at<double>(1),
                        tvec_transposed.at<double>(2));
        }
        if (rms_transposed < rms_direct) {
            observation.corners = reordered;
            observation.rvec = rvec_transposed;
            observation.tvec = tvec_transposed;
            observation.reprojection_rms_px = rms_transposed;
        } else {
            observation.rvec = rvec_direct;
            observation.tvec = tvec_direct;
            observation.reprojection_rms_px = rms_direct;
        }
        return true;
    }
} // namespace calibration
