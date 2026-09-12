#include "../include/pnp_solver.hpp"

namespace energy {

solver::solver(const SolvePnPconfig& config)
{
    energy_points_[0] = cv::Point3f(0, -ENERGY_WIDTH / 2.0, ENERGY_LENGTH / 2.0);
    energy_points_[1] = cv::Point3f(0,  ENERGY_WIDTH / 2.0, ENERGY_LENGTH / 2.0);
    energy_points_[2] = cv::Point3f(0,  ENERGY_WIDTH / 2.0, -ENERGY_LENGTH / 2.0);
    energy_points_[3] = cv::Point3f(0, -ENERGY_WIDTH / 2.0, -ENERGY_LENGTH / 2.0);

    camera_matrix_ = cv::Mat(3, 3, CV_64F,
                             const_cast<double*>(config.camera_matrix.data())).clone();
    dist_coeffs_ = cv::Mat(config.distortion_coefficients, true);
}

SolvePnPresult solver::getSolveResult(const EnergyBlade& blade)
{
    std::vector<cv::Point2f> img_pts;
    img_pts.reserve(4);
    img_pts.emplace_back(blade.object_points[0]);
    img_pts.emplace_back(blade.object_points[1]);
    img_pts.emplace_back(blade.object_points[2]);
    img_pts.emplace_back(blade.object_points[3]);

    std::vector<cv::Point3f> obj_pts(energy_points_, energy_points_ + 4);

    bool success = cv::solvePnP(obj_pts, img_pts,
                                 camera_matrix_, dist_coeffs_,
                                 rvec_, tvec_, false,
                                 cv::SOLVEPNP_IPPE);

    SolvePnPresult result;
    result.tvec = tvec_.clone();
    result.rvec = rvec_.clone();
    result.distance = success ? cv::norm(tvec_) : 0.0;
    if (!success) {
        std::cerr << "SolvePnP Failed!" << std::endl;
    }
    return result;
}

} // namespace energy
