#include "simulator.h"

#include <fstream>
#include <iomanip>
#include <random>

namespace rm
{
    std::vector<SimulationFrame> runSimulation(const SimulationConfig& cfg, const std::string& csv_path)
    {
        std::mt19937 rng(cfg.seed);
        std::normal_distribution<double> noise(0.0, cfg.pixel_noise_std);
        std::normal_distribution<double> box_jit(0.0, cfg.box_jitter_px);
        std::normal_distribution<double> ts_jit(0.0, cfg.timeline_jitter_std);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::uniform_real_distribution<double> spike(-cfg.outlier_px, cfg.outlier_px);
        std::vector<SimulationFrame> frames;
        frames.reserve(cfg.num_frames);

        std::ofstream csv;
        if (!csv_path.empty())
        {
            csv.open(csv_path);
            csv << "t,top_u,top_v,bot_u,bot_v,center_x,center_y,center_z,yaw,omega\n";
            csv << std::fixed << std::setprecision(6);
        }

        for (int i = 0; i < cfg.num_frames; ++i)
        {
            const double t = i * cfg.dt;
            if (cfg.dropout_frac > 0.0 && uni(rng) < cfg.dropout_frac) continue;
            const double theta = cfg.theta0 + cfg.omega0 * t + 0.5 * cfg.omega_accel * t * t;
            const double omega = cfg.omega0 + cfg.omega_accel * t;
            const Vec3 C = cfg.center_cam + cfg.drift_velocity * t + 0.5 * cfg.drift_accel * t * t;
            const double pitch_t = cfg.gimbal_pitch + cfg.gimbal_pitch_rate * t;

            Vec3 top, bot;
            barEndpointsCamera(C, cfg.lightbar_h, theta, cfg.pitch_install, cfg.radius,
                               cfg.lightbar_len, pitch_t, top, bot);
            SimulationFrame f;
            f.timestamp = t + (cfg.timeline_jitter_std > 0 ? ts_jit(rng) : 0.0);
            f.top_px = project(top, cfg.K);
            f.bottom_px = project(bot, cfg.K);
            if (cfg.box_jitter_px > 0.0)
            {
                const Vec2 bj(box_jit(rng), box_jit(rng));
                f.top_px += bj; f.bottom_px += bj;
            }
            if (cfg.enable_noise && cfg.pixel_noise_std > 0.0)
            {
                f.top_px += Vec2(noise(rng), noise(rng));
                f.bottom_px += Vec2(noise(rng), noise(rng));
            }
            if (cfg.outlier_px > 0.0 && uni(rng) < cfg.outlier_frac)
            {
                f.top_px += Vec2(spike(rng), spike(rng));
            }
            f.center_cam_gt = C;
            f.yaw_gt = theta;
            f.omega_gt = omega;
            f.gimbal_pitch_gt = pitch_t;
            frames.push_back(f);

            if (csv)
                csv << t << ',' << f.top_px.x() << ',' << f.top_px.y() << ','
                    << f.bottom_px.x() << ',' << f.bottom_px.y() << ','
                    << C.x() << ',' << C.y() << ',' << C.z() << ',' << theta << ',' << omega << '\n';
        }
        return frames;
    }
}  // namespace rm
