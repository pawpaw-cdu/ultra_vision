#include "auto_buff/rune_types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace auto_aim::energy
{
    FanBlade::FanBlade(const std::vector<cv::Point2f>& keypoints, cv::Point2f keypoints_center,
                       FanBladeType blade_type)
        : center(keypoints_center), points(keypoints), type(blade_type)
    {
    }

    FanBlade::FanBlade(FanBladeType blade_type) : type(blade_type)
    {
        if (blade_type != Unlit) {
            throw std::invalid_argument("FanBlade: only unlit placeholder may be default-built");
        }
    }

    PowerRune::PowerRune(std::vector<FanBlade>& blades, cv::Point2f center,
                         std::optional<PowerRune> last_powerrune)
        : r_center(center), light_num(static_cast<int>(blades.size()))
    {
        if (blades.empty()) {
            unsolvable_ = true;
            return;
        }

        // ---- pick the current target blade -------------------------------------
        if (light_num == 1) {
            // Only one blade is lit, so it is the target.
            blades[0].type = Target;
        } else if (last_powerrune.has_value() &&
                   light_num == last_powerrune->light_num) {
            // The same blades are still lit: the target is the one closest to
            // the previous target.
            const cv::Point2f last_target_center = last_powerrune->fanblades[0].center;
            auto best = blades.begin();
            float best_distance = cv::norm(blades[0].center - last_target_center);
            for (auto it = blades.begin(); it != blades.end(); ++it) {
                const float distance = cv::norm(it->center - last_target_center);
                if (distance < best_distance) {
                    best_distance = distance;
                    best = it;
                }
            }
            best->type = Target;
            std::iter_swap(blades.begin(), best);
        } else if (last_powerrune.has_value() &&
                   light_num == last_powerrune->light_num + 1) {
            // A blade just lit up: it is the one farthest from every blade that
            // was already lit.
            const auto& last_blades = last_powerrune->fanblades;
            auto best = blades.begin();
            float best_min_distance = -1.0f;
            for (auto it = blades.begin(); it != blades.end(); ++it) {
                float min_distance = std::numeric_limits<float>::max();
                for (const auto& last_blade : last_blades) {
                    if (last_blade.type == Unlit) continue;
                    const float distance = cv::norm(it->center - last_blade.center);
                    min_distance = std::min(min_distance, distance);
                }
                if (min_distance > best_min_distance) {
                    best_min_distance = min_distance;
                    best = it;
                }
            }
            best->type = Target;
            std::iter_swap(blades.begin(), best);
        } else {
            unsolvable_ = true;
            return;
        }

        // ---- angular position of every lit blade relative to the target -------
        const double target_angle = atan_angle(blades[0].center);
        for (auto& blade : blades) {
            blade.angle = atan_angle(blade.center) - target_angle;
            if (blade.angle < -1e-3) blade.angle += 2.0 * CV_PI;
        }

        std::sort(blades.begin(), blades.end(),
                  [](const FanBlade& lhs, const FanBlade& rhs) {
                      return lhs.angle < rhs.angle;
                  });

        // ---- place the lit blades on the fixed five-slot lattice -------------
        const double step = 2.0 * CV_PI / 5.0;
        fanblades.clear();
        fanblades.reserve(5);
        for (int slot = 0, lit = 0; slot < 5 && lit < static_cast<int>(blades.size()); ++slot) {
            const double slot_angle = slot * step;
            if (std::abs(blades[lit].angle - slot_angle) < CV_PI / 5.0) {
                fanblades.emplace_back(blades[lit++]);
            } else {
                fanblades.emplace_back(FanBlade(Unlit));
            }
        }
        while (fanblades.size() < 5) fanblades.emplace_back(FanBlade(Unlit));
    }

    double PowerRune::atan_angle(cv::Point2f point) const
    {
        const cv::Point2f difference = point - r_center;
        const double angle = std::atan2(difference.y, difference.x);
        return angle >= 0.0 ? angle : angle + 2.0 * CV_PI;
    }
} // namespace auto_aim::energy
