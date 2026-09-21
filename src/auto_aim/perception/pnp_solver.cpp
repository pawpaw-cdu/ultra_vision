#include "pnp_solver.hpp"
#include <iostream>

namespace auto_aim
{
    cv::Mat CAMERA_MATRIX = (cv::Mat_<double>(3,3) << 1286.0, 0.0, 645.0,
                                                     0.0, 1288.0, 483.0,
                                                     0.0, 0.0, 1.0);
    cv::Mat DIST_COEFFS = cv::Mat::zeros(5, 1, CV_64F);

    std::vector<cv::Point3f> getArmor3DPoints(int type, float small_width, float large_width, float height)
    {
        float half_w = (type == 0) ? small_width / 2.0f : large_width / 2.0f;
        float half_h = height / 2.0f;
        return {
            cv::Point3f(0,  half_w, -half_h), // left_bottom
            cv::Point3f(0,  half_w,  half_h), // left_top
            cv::Point3f(0, -half_w,  half_h), // right_top
            cv::Point3f(0, -half_w, -half_h)  // right_bottom
        };
    }

    bool solveArmorPnP(auto_aim::Armor& armor,
                       const cv::Mat& camera_matrix,
                       const cv::Mat& dist_coeffs,
                       float small_width, float large_width, float height)
    {
        int type = (armor.armor_type == auto_aim::small) ? 0 : 1;
        std::vector<cv::Point3f> obj_points = getArmor3DPoints(type, small_width, large_width, height);

        std::vector<cv::Point2f> img_points = {
            armor.left.bottom, armor.left.top,
            armor.right.top, armor.right.bottom
        };

        return cv::solvePnP(obj_points, img_points, camera_matrix, dist_coeffs,
                            armor.rvec, armor.tvec, false, cv::SOLVEPNP_IPPE);
    }
}