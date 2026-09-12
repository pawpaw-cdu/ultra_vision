#ifndef PNP_SOLVER_HPP
#define PNP_SOLVER_HPP

#include <opencv2/opencv.hpp>
#include "types.hpp"
#include "config.hpp"

namespace energy {

#define ENERGY_LENGTH 382   // 单位：mm
#define ENERGY_WIDTH 320

class solver
{
public:
    explicit solver(const SolvePnPconfig& config);
    ~solver() = default;

    SolvePnPresult getSolveResult(const EnergyBlade& blade);

private:
    cv::Point3f energy_points_[4];
    cv::Mat camera_matrix_, dist_coeffs_;
    cv::Mat tvec_, rvec_;
};

} // namespace energy

#endif
