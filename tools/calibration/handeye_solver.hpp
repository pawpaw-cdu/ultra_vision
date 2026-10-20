#ifndef ULTRA_VISION_CALIBRATION_HANDEYE_SOLVER_HPP
#define ULTRA_VISION_CALIBRATION_HANDEYE_SOLVER_HPP

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace calibration
{
    struct Sample
    {
        double yaw = 0.0;       // gimbal absolute yaw (rad, unwrapped)
        double pitch = 0.0;     // gimbal absolute pitch (rad)
        cv::Mat rvec;           // board pose in camera frame
        cv::Mat tvec;
        double time_s = 0.0;
    };

    struct FitOptions
    {
        bool fit_yaw_scale = false;
        bool fit_pitch_scale = false;
        double t_max = 0.30;
        double rot_max = 0.5;
    };

    struct Fit
    {
        cv::Matx33d R = cv::Matx33d::eye();  // camera -> gimbal
        cv::Vec3d t{0.0, 0.0, 0.0};          // metres
        double yaw_scale = 1.0;
        double pitch_scale = 1.0;
        double residual_mm = 0.0;
        cv::Vec3d t_sigma{0.0, 0.0, 0.0};
        double cross_validate_mm = -1.0;
        bool ok = false;
    };

    double wrapPi(double value);
    double unwrapYaw(double yaw, double reference);
    void unwrapSamples(std::vector<Sample>& samples);

    bool solve(const std::vector<Sample>& samples, const FitOptions& options, Fit& fit);
    double scanYawScale(const std::vector<Sample>& samples, const FitOptions& options,
                        bool verbose);
    double crossValidate(const std::vector<Sample>& samples, const FitOptions& options,
                         double* in_sample_mm = nullptr);
    std::string sanityProblem(const std::vector<Sample>& samples, const Fit& fit);
    void printFit(const std::vector<Sample>& samples, const Fit& fit, const std::string& tag);
    void writeYaml(const std::string& output, const std::vector<Sample>& samples,
                   const Fit& fit, const std::string& comment = {});
    bool saveIfGood(const std::string& output, const std::vector<Sample>& samples,
                    const Fit& fit);
} // namespace calibration

#endif // ULTRA_VISION_CALIBRATION_HANDEYE_SOLVER_HPP
