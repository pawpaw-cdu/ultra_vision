#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <array>
#include <chrono>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "estimation/rotation_rate_estimator.hpp"
#include "armor_ekf.hpp"
#include "common/types.hpp"

namespace auto_aim
{
    enum class TrackerState { DETECTING, TRACKING, TEMP_LOST, LOST };

    struct CameraPose {
        bool valid = true;
        bool timestamp_valid = false;
        double timestamp = 0.0;
        double yaw = 0.0;
        double pitch = 0.0;
        // world <- base. Simulator uses identity; a real robot can fill this
        // from IMU so the target EKF stays in the world frame.
        std::array<std::array<double, 3>, 3> base_to_world{{
            {{1.0, 0.0, 0.0}},
            {{0.0, 1.0, 0.0}},
            {{0.0, 0.0, 1.0}}
        }};
    };

    struct TrackerConfig {
        int min_detect_frames = 3;
        int max_lost_frames = 10;
        int reacquire_frames = 3;
        double max_match_distance = 0.5;
        double reacquire_center_error = 0.25;
        double reacquire_max_dt = 0.15;
        double max_camera_pose_dt = 0.05;

        RotationRateEstimatorConfig rotation_rate;
        // Fusion weights for the external radial-angle rate estimate. It is
        // applied as a gated scalar measurement of the EKF rate state.
        //
        // The external estimate is deliberately down-weighted: on a 3 rad/s
        // small-gyro run it proved biased low (|omega| median 1.48 vs a true
        // 3.0), while the EKF's own estimate from the armor_yaw sequence
        // reached |omega| median 2.49. Treating it as a strong measurement made
        // the fused rate worse, so it is kept as a weak prior only.
        double omega_measure_noise = 100.0;
        double omega_nis_threshold = 9.0;

        // These dimensions are kept for compatibility with the existing YAML.
        int state_dim = 6;
        int measure_dim = 3;
        double process_noise_pos = 1.0;
        double process_noise_vel = 400.0;
        double measure_noise = 0.01;
        double init_error_cov = 1.0;
        double default_dt = 0.033;

        // Parameters of the circular armor EKF.
        double armor_radius = 0.21;
        double y_noise_mult = 30.0;
        double p_cov_min = 1e-3;
        double angle_measure_noise = 0.04;
        double bearing_measure_noise = 0.0025;
        double pitch_measure_noise = 0.005;
        double range_noise_distance_scale = 0.1;
        double ypd_nis_threshold = 9.488;
        double max_angle_error = 0.70;
        double angle_match_weight = 0.08;
        double position_nis_threshold = 7.815;
        double angle_nis_threshold = 3.841;
        // Per-armor vertical center offsets. They remain zero for a symmetric
        // real robot model and can be calibrated for a simulator vehicle.
        std::array<double, 4> armor_y_offsets{{0.0, 0.0, 0.0, 0.0}};
    };

    class Tracker
    {
    public:
        explicit Tracker(const TrackerConfig& cfg);
        ~Tracker() = default;

        void init(const Armor& armor, double timestamp);
        void init(const std::vector<Armor>& armors, double timestamp);
        Armor predict(double timestamp);
        bool update(const Armor& armor, double timestamp);
        bool update(const std::vector<Armor>& armors, double timestamp);
        // Advance the estimate to `timestamp` without registering a missed
        // detection. The asynchronous detector simply has no new result for
        // some frames, and those must not count towards the loss timeout.
        void predictTo(double timestamp);
        void lostUpdate();

        TrackerState getState() const { return state_; }
        const Armor& getLatestArmor() const { return latest_armor_; }
        const Armor& getLastObservedArmor() const { return last_observed_armor_; }
        int getLastObservedPlateId() const { return last_observed_plate_id_; }
        bool hasObservation() const { return has_observation_; }
        double getConfidence() const { return confidence_; }
        int getLostCount() const { return lost_count_; }

        cv::Mat getTargetCenter() const;
        std::array<double, 3> getTargetCenterArray() const;
        std::array<double, 3> getTargetVelocity() const;
        std::array<double, 3> getTargetCenterWorldArray() const;
        std::array<double, 3> getTargetVelocityWorld() const;
        std::array<cv::Mat, 4> getEstimatedArmorPositions() const;
        double getYaw() const;
        double getYawWorld() const { return ekf_.getYaw(); }
        // The EKF owns the rate state: the external estimator is only one of
        // its measurements, so both the selector and the telemetry read the
        // fused value from the filter.
        double getOmega() const { return ekf_.getOmega(); }
        double getRotationRateOmega() const {
            return rotation_rate_estimator_.omega();
        }
        double getEkfOmega() const { return ekf_.getOmega(); }
        bool hasRotationRate() const { return rotation_rate_estimator_.valid(); }
        double getArmorRadius() const { return cfg_.armor_radius; }
        double getLastMeasurementWeight() const { return last_measurement_weight_; }

        std::array<double, 3> worldToCameraPoint(
            const std::array<double, 3>& point) const;
        double worldToCameraYaw(double yaw) const;

        // The EKF state is maintained in the world frame. Set the timestamped
        // camera pose used by the next observation before update().
        void setCameraPose(const CameraPose& pose);
        void setCameraAngles(double yaw, double pitch);

    private:
        TrackerConfig cfg_;
        ArmorEKF ekf_;
        RotationRateEstimator rotation_rate_estimator_;
        double last_timestamp_;
        TrackerState state_;
        int detect_count_;
        int lost_count_;
        double confidence_;
        int plate_id_;
        double last_measurement_weight_;
        Armor latest_armor_;
        Armor last_observed_armor_;
        int last_observed_plate_id_;
        bool has_observation_;
        CameraPose camera_pose_;

        Armor toWorld(const Armor& armor) const;
        Vec<3> worldToCameraVector(const Vec<3>& value) const;
        cv::Mat worldToCamera(const Vec<3>& value) const;

        void predictStateTo(double timestamp);
        bool tryReacquire(const std::vector<const Armor*>& observations,
                          double timestamp);
        int selectPlateId(const Armor& armor) const;
        double distanceToPlate(const cv::Mat& tvec, int plate_id) const;
        double matchCost(const Armor& armor, int plate_id) const;
        void updateRotationRate(const Armor& armor, double timestamp);
    };
}

#endif
