#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <array>
#include <chrono>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "armor_ekf.hpp"
#include "common/types.hpp"

namespace auto_aim
{
    enum class TrackerState { DETECTING, TRACKING, TEMP_LOST, LOST };

    struct TrackerConfig {
        int min_detect_frames = 3;
        int max_lost_frames = 10;
        double max_match_distance = 0.5;

        // These dimensions are kept for compatibility with the existing YAML.
        int state_dim = 6;
        int measure_dim = 3;
        double process_noise_pos = 1.0;
        double process_noise_vel = 2.0;
        double measure_noise = 0.01;
        double init_error_cov = 1.0;
        double default_dt = 0.033;

        // Parameters of the circular armor EKF.
        double armor_radius = 0.21;
        double y_noise_mult = 30.0;
        double p_cov_min = 1e-3;
        double angle_measure_noise = 0.04;
        double max_angle_error = 0.70;
        double angle_match_weight = 0.08;
        double position_nis_threshold = 7.815;
        double angle_nis_threshold = 3.841;
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
        void lostUpdate();

        TrackerState getState() const { return state_; }
        const Armor& getLatestArmor() const { return latest_armor_; }
        const Armor& getLastObservedArmor() const { return last_observed_armor_; }
        int getLastObservedPlateId() const { return last_observed_plate_id_; }
        bool hasObservation() const { return has_observation_; }
        double getConfidence() const { return confidence_; }
        int getLostCount() const { return lost_count_; }

        cv::Mat getTargetCenter() const;
        std::array<cv::Mat, 4> getEstimatedArmorPositions() const;
        double getYaw() const { return ekf_.getYaw(); }
        double getOmega() const { return ekf_.getOmega(); }
        double getLastMeasurementWeight() const { return last_measurement_weight_; }

    private:
        struct ChassisFit {
            Vec<3> center;
            double yaw = 0.0;
            std::array<int, 4> plate_ids{{-1, -1, -1, -1}};
            double error = 0.0;
            bool valid = false;
        };

        TrackerConfig cfg_;
        ArmorEKF ekf_;
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

        void predictStateTo(double timestamp);
        bool fitChassis(const std::vector<const Armor*>& observations,
                        ChassisFit& fit) const;
        bool updateFromFit(const std::vector<const Armor*>& observations,
                           const ChassisFit& fit, int& best_armor_index,
                           int& best_plate_id);
        int selectPlateId(const Armor& armor) const;
        double distanceToPlate(const cv::Mat& tvec, int plate_id) const;
        double matchCost(const Armor& armor, int plate_id) const;
    };
}

#endif
