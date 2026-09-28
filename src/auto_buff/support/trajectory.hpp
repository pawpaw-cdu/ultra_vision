#ifndef AUTO_AIM_ENERGY_SUPPORT_TRAJECTORY_HPP
#define AUTO_AIM_ENERGY_SUPPORT_TRAJECTORY_HPP

// Ballistic solution for the energy-rune aimer (no air drag).
//
// Ported from sp_vision_25 `tools/trajectory.{hpp,cpp}`; gravity is now a
// parameter so the simulator value (9.81) and the measured real-robot value
// (9.7833) can both be configured.

#include <cmath>

namespace auto_aim::energy
{
    struct Trajectory
    {
        bool unsolvable = true;
        double fly_time = 0.0;
        double pitch = 0.0; // positive is upwards

        // v0: bullet speed (m/s), d: horizontal distance (m), h: height (m).
        Trajectory(double v0, double d, double h, double gravity = 9.81)
        {
            const double a = gravity * d * d / (2.0 * v0 * v0);
            const double b = -d;
            const double c = a + h;
            const double delta = b * b - 4.0 * a * c;
            if (delta < 0.0 || std::abs(a) < 1e-12) {
                unsolvable = true;
                return;
            }

            const double tan_1 = (-b + std::sqrt(delta)) / (2.0 * a);
            const double tan_2 = (-b - std::sqrt(delta)) / (2.0 * a);
            const double pitch_1 = std::atan(tan_1);
            const double pitch_2 = std::atan(tan_2);
            const double t_1 = d / (v0 * std::cos(pitch_1));
            const double t_2 = d / (v0 * std::cos(pitch_2));

            unsolvable = false;
            pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
            fly_time = (t_1 < t_2) ? t_1 : t_2;
        }
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_SUPPORT_TRAJECTORY_HPP
