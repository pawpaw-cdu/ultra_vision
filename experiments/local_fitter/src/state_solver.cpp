#include "state_solver.h"
#include <cmath>

namespace rm
{
    VehicleState solveVehicleState(const FitResult& fit, double timestamp)
    {
        VehicleState s;
        s.timestamp = timestamp;
        s.center_cam = fit.center_cam;
        s.radius = fit.R_used;
        s.yaw = fit.yawAt(timestamp);
        s.pitch_install = fit.alpha_used;
        s.angular_velocity = fit.omega + fit.beta * timestamp;
        s.valid = fit.success;
        s.quality = fit.reprojection_rmse;
        return s;
    }

    std::vector<ArmorPose> extrapolateArmors(const VehicleState& state, int num_armors)
    {
        std::vector<ArmorPose> out;
        const double dtheta = 2.0 * M_PI / std::max(1, num_armors);
        for (int k = 0; k < num_armors; ++k)
        {
            const double theta = state.yaw + k * dtheta;
            ArmorPose a;
            a.armor_id = k;
            a.center_cam = state.center_cam +
                rotationVehicleToCamera(0.0) * Vec3(state.radius * std::cos(theta),
                                                    state.radius * std::sin(theta), 0.0);
            out.push_back(a);
        }
        return out;
    }
}  // namespace rm
