#ifndef AUTO_AIM_YPD_OBSERVATION_HPP
#define AUTO_AIM_YPD_OBSERVATION_HPP

// YPD（球坐标：yaw/pitch/distance/armor_yaw）观测。从 armor_ekf.hpp 抽出。

namespace auto_aim
{

// YPD measurement: bearing yaw, pitch, range and the armor radial angle.
// Noise values are variances in rad^2, rad^2, m^2 and rad^2 respectively.
struct YpdObservation {
    int plate_id = -1;
    double yaw = 0.0;
    double pitch = 0.0;
    double distance = 0.0;
    double armor_yaw = 0.0;
    double yaw_noise = 2.5e-3;
    double pitch_noise = 5.0e-3;
    double distance_noise = 1.0e-2;
    /// @brief `armor_yaw`（观测板的朝向 → 底盘 yaw 的约束）量测噪声**方差**（rad²）。
    /// 4e-2 rad² 相当于 σ≈11.5°，等于告诉滤波器"板的朝向几乎不可信" —— 而单板工况下
    /// 底盘 yaw / 中心沿切向就是**唯一不可辨识**的方向，把这个观测压到没用，滤波器就只能
    /// 沿切向漂（离线实测：模型-观测偏差 93% 都在切向，137 mm）。PnP 给的板朝向实际精度是
    /// 1~2°，所以这里取 (1.5°)²≈7e-4 rad²。（old: 4.0e-2）
    double armor_yaw_noise = 7.0e-4;
};
}

#endif // AUTO_AIM_YPD_OBSERVATION_HPP
