#pragma once
#include "trajectory_fitter.h"
#include "types.h"
#include <vector>

namespace rm
{
    VehicleState solveVehicleState(const FitResult& fit, double timestamp);

    struct ArmorPose { int armor_id; Vec3 center_cam; };
    std::vector<ArmorPose> extrapolateArmors(const VehicleState& state, int num_armors = 4);
}  // namespace rm
