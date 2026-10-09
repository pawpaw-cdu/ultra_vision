#pragma once
#include "simulator.h"
#include "trajectory_fitter.h"
#include <optional>
#include <string>
#include <vector>

namespace rm
{
    void visualizeResults(const std::vector<SimulationFrame>& frames,
                          const std::optional<FitResult>& fit,
                          const CameraIntrinsics& K, double gimbal_pitch, double alpha, double R,
                          double L, const std::string& output_path);
}  // namespace rm
