#include "simulator.h"

#include <algorithm>
#include <cmath>
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

    std::vector<MultiFrame> runMultiSimulation(const SimulationConfig& cfg, const std::string& csv_path)
    {
        std::mt19937 rng(cfg.seed);
        std::normal_distribution<double> noise(0.0, cfg.pixel_noise_std);
        std::normal_distribution<double> box_jit(0.0, cfg.box_jitter_px);
        std::normal_distribution<double> ts_jit(0.0, cfg.timeline_jitter_std);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::uniform_real_distribution<double> spike(-cfg.outlier_px, cfg.outlier_px);
        const int np = std::max(1, cfg.num_plates);
        std::vector<MultiFrame> frames;
        frames.reserve(cfg.num_frames);

        std::ofstream csv;
        if (!csv_path.empty())
        {
            csv.open(csv_path);
            csv << "t,armor_id,lt_u,lt_v,lb_u,lb_v,rt_u,rt_v,rb_u,rb_v,"
                   "center_x,center_y,center_z,yaw,omega,radius\n";
            csv << std::fixed << std::setprecision(6);
        }

        for (int i = 0; i < cfg.num_frames; ++i)
        {
            const double t = i * cfg.dt;
            if (cfg.dropout_frac > 0.0 && uni(rng) < cfg.dropout_frac) continue;
            const double theta = cfg.theta0 + cfg.omega0 * t + 0.5 * cfg.omega_accel * t * t;
            const double omega = cfg.omega0 + cfg.omega_accel * t;
            const double R = (cfg.radius_step_time >= 0.0 && t >= cfg.radius_step_time)
                                 ? cfg.radius_after : cfg.radius;
            const Vec3 C = cfg.center_cam + cfg.drift_velocity * t + 0.5 * cfg.drift_accel * t * t;
            const double pitch_t = cfg.gimbal_pitch + cfg.gimbal_pitch_rate * t;
            const Mat3 rcv = rotationVehicleToCamera(pitch_t);

            MultiFrame mf;
            mf.timestamp = t + (cfg.timeline_jitter_std > 0 ? ts_jit(rng) : 0.0);
            mf.center_cam_gt = C;
            mf.yaw_gt = theta;
            mf.omega_gt = omega;
            mf.gimbal_pitch_gt = pitch_t;
            mf.radius_gt = R;

            for (int k = 0; k < np; ++k)
            {
                const double theta_k = theta + k * (2.0 * M_PI / np);
                // 自遮挡：板法线朝向相机的角度门控
                if (cfg.gate_visibility)
                {
                    const Vec3 n_v(std::cos(theta_k), std::sin(theta_k), 0.0);
                    const Vec3 n_c = rcv * n_v;
                    const Vec3 plate_center_c = rcv * (R * n_v) + C;
                    const Vec3 to_cam = -plate_center_c;
                    const double denom = to_cam.norm() * n_c.norm();
                    const double cosang = denom > 1e-9 ? n_c.dot(to_cam) / denom : -1.0;
                    if (cosang < std::cos(cfg.visibility_half_angle)) continue;
                }

                Vec3 corners[4];
                plateCornersCamera(C, cfg.lightbar_h, theta_k, cfg.pitch_install, R,
                                   cfg.lightbar_len, cfg.plate_width, pitch_t, corners);
                PlateObservation po;
                po.armor_id = k;
                Vec2 bj(0, 0);
                if (cfg.box_jitter_px > 0.0) bj = Vec2(box_jit(rng), box_jit(rng));
                double worst_spike = 0.0;
                for (int c = 0; c < 4; ++c)
                {
                    Vec2 px = project(corners[c], cfg.K) + bj;
                    if (cfg.enable_noise && cfg.pixel_noise_std > 0.0)
                        px += Vec2(noise(rng), noise(rng));
                    if (cfg.outlier_px > 0.0 && uni(rng) < cfg.outlier_frac)
                    {
                        const Vec2 sp(spike(rng), spike(rng));
                        px += sp;
                        worst_spike = std::max(worst_spike, sp.norm());
                    }
                    po.corners_px[c] = px;
                }
                po.confidence = worst_spike > 0.0 ? 0.3 : 1.0;
                mf.plates.push_back(po);
                mf.any_visible = true;

                if (csv)
                    csv << t << ',' << k
                        << ',' << po.corners_px[0].x() << ',' << po.corners_px[0].y()
                        << ',' << po.corners_px[1].x() << ',' << po.corners_px[1].y()
                        << ',' << po.corners_px[2].x() << ',' << po.corners_px[2].y()
                        << ',' << po.corners_px[3].x() << ',' << po.corners_px[3].y()
                        << ',' << C.x() << ',' << C.y() << ',' << C.z()
                        << ',' << theta << ',' << omega << ',' << R << '\n';
            }
            frames.push_back(std::move(mf));
        }
        return frames;
    }
}  // namespace rm
