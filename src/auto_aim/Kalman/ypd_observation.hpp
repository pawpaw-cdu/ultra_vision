#ifndef AUTO_AIM_YPD_OBSERVATION_HPP
#define AUTO_AIM_YPD_OBSERVATION_HPP

// YPD（球坐标：yaw/pitch/distance/armor_yaw）观测。从 armor_ekf.hpp 抽出。

namespace auto_aim
{

struct YpdObservation {
    int plate_id = -1;
    double yaw = 0.0;
    double pitch = 0.0;
    double distance = 0.0;
    double armor_yaw = 0.0;
    double yaw_noise = 2.5e-3;
    double pitch_noise = 5.0e-3;
    double distance_noise = 1.0e-2;
    double armor_yaw_noise = 7.0e-4;
};
}

#endif // AUTO_AIM_YPD_OBSERVATION_HPP
