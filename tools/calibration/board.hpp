#ifndef ULTRA_VISION_CALIBRATION_BOARD_HPP
#define ULTRA_VISION_CALIBRATION_BOARD_HPP

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace calibration
{
    struct BoardConfig
    {
        std::string pattern = "chessboard";  // chessboard | circles
        int cols = 11;
        int rows = 8;
        double square_mm = 15.0;
        std::string grid = "symmetric";
    };

    struct BoardObservation
    {
        std::vector<cv::Point2f> corners;
        cv::Mat rvec;
        cv::Mat tvec;
        double reprojection_rms_px = -1.0;
    };

    BoardConfig loadBoardConfig(const std::string& config_dir);
    std::vector<cv::Point3f> boardObjectPoints(const BoardConfig& board,
                                               double unit_scale = 1e-3);

    bool detectBoardCorners(const cv::Mat& image, const BoardConfig& board,
                            std::vector<cv::Point2f>& corners,
                            bool refine_subpixel = true,
                            bool allow_classic_fallback = true);

    bool detectBoard(const cv::Mat& image, const BoardConfig& board,
                     const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                     BoardObservation& observation,
                     bool allow_classic_fallback = true);
} // namespace calibration

#endif // ULTRA_VISION_CALIBRATION_BOARD_HPP
