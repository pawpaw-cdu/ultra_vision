#include "auto_buff/rune_diag.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>

#include "common/standard_clock.hpp"

namespace auto_aim::energy::rune_diag
{
    namespace
    {
        // Pulls one `key=value` field out of a simulator telemetry line.
        std::string telemetryField(const std::string& line, const std::string& key)
        {
            const std::string needle = key + "=";
            const auto position = line.find(needle);
            if (position == std::string::npos) return {};
            const auto start = position + needle.size();
            const auto end = line.find(' ', start);
            return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
        }

        double percentile(std::vector<double> values, double fraction)
        {
            if (values.empty()) return 0.0;
            std::sort(values.begin(), values.end());
            return values[static_cast<std::size_t>(fraction * (values.size() - 1))];
        }
    } // namespace

    bool debugEnabled() { return std::getenv("ULTRA_VISION_RUNE_DEBUG") != nullptr; }

    void StageProfiler::add(const char* stage, double milliseconds)
    {
        if (!enabled_) return;
        for (auto& entry : stages_) {
            if (entry.first == stage) {
                entry.second.first += milliseconds;
                entry.second.second += 1;
                return;
            }
        }
        stages_.emplace_back(stage, std::make_pair(milliseconds, 1));
    }

    void StageProfiler::flushAndReset() const
    {
        if (!enabled_ || stages_.empty()) return;
        std::cout << "  profile ms/frame:";
        for (const auto& entry : stages_) {
            const int count = std::max(1, entry.second.second);
            std::cout << ' ' << entry.first << '=' << std::fixed << std::setprecision(1)
                      << entry.second.first / count;
        }
        std::cout << std::endl;
        stages_.clear();
    }

    void GimbalTrace::record(double yaw, double pitch, double now)
    {
        if (valid_) {
            const double dt = now - last_time_;
            if (dt > 0.0 && dt < 0.2) {
                const double dyaw = std::abs(std::atan2(std::sin(yaw - last_yaw_),
                                                        std::cos(yaw - last_yaw_)));
                const double step = std::max(dyaw, std::abs(pitch - last_pitch_));
                steps_deg_.push_back(step * 180.0 / CV_PI);
                rates_deg_s_.push_back(step * 180.0 / CV_PI / dt);
            }
        }
        last_yaw_ = yaw;
        last_pitch_ = pitch;
        last_time_ = now;
        valid_ = true;
    }

    void printFrameDebug(const RuneFrameDebug& debug)
    {
        const RuneDetector& detector = debug.detector;
        const PowerRune* rune = debug.rune;
        const RuneCommand& command = debug.command;
        const double deg = 180.0 / CV_PI;
        std::cout << "[frame] " << debug.frame_index
                  << " st=" << runeStatusName(detector.status())
                  << " det=" << (rune != nullptr ? 1 : 0)
                  << " solved=" << (rune != nullptr && rune->solved ? 1 : 0);
        if (rune != nullptr) {
            std::cout << " obs_yaw=" << rune->ypd_in_world[0] * deg
                      << " obs_pitch=" << rune->ypd_in_world[1] * deg
                      << " obs_dis=" << rune->ypd_in_world[2]
                      << " r_c=(" << rune->r_center.x << "," << rune->r_center.y << ")";
        }
        std::cout << " pose=(" << debug.commanded_yaw * deg << "," << debug.commanded_pitch * deg
                  << ")"
                  << " cmd=(" << command.yaw * deg << "," << command.pitch * deg << ")"
                  << " fly=" << command.fly_time
                  << " cls=" << detector.targetClass()
                  << " shoot=" << (command.shoot ? 1 : 0)
                  << " ctl=" << (command.control ? 1 : 0) << std::endl;
    }

    void printFireGateDebug(const RuneFireGateDebug& gate)
    {
        std::cout << "  [fire] shoot=" << gate.shoot
                  << " switched=" << gate.switched
                  << " aim_ready=" << gate.aim_ready
                  << " hold=" << gate.hold
                  << " engageable=" << gate.engageable
                  << " window_open=" << gate.window_open << std::endl;
    }

    void printFireSuppressed(const RuneCommand& command)
    {
        std::cout << "[fire suppressed] yaw=" << command.yaw << " pitch=" << command.pitch
                  << std::endl;
    }

    void printFpsLine(double fps, const RuneDetector& detector, const RuneTarget& target,
                      const RuneCommand& command, long long skipped_frames)
    {
        std::cout << "Energy rune FPS: " << fps
                  << ", nn_ms: " << detector.latencyMs()
                  << ", lost: " << detector.lostCount()
                  << ", solvable: " << !target.isUnsolvable();
        if (!target.isUnsolvable()) {
            std::cout << ", dis: " << target.ekfX()[3] << ", roll_deg: "
                      << target.ekfX()[5] * 180.0 / CV_PI << ", spd_deg: "
                      << target.ekfX()[6] * 180.0 / CV_PI;
        }
        std::cout << ", control: " << command.control << ", fire: " << command.shoot;
        if (skipped_frames > 0) std::cout << ", dropped: " << skipped_frames;
        std::cout << std::endl;
    }

    namespace
    {
        void drawObservation(cv::Mat& image, const PowerRune& rune)
        {
            for (const auto& blade : rune.fanblades) {
                if (blade.type == Unlit) continue;
                for (std::size_t i = 0; i < blade.points.size(); ++i) {
                    cv::circle(image, blade.points[i], 2, cv::Scalar(255, 255, 0), -1);
                }
                for (int i = 0; i < 4; ++i) {
                    cv::line(image, blade.points[static_cast<std::size_t>(i)],
                             blade.points[static_cast<std::size_t>((i + 1) % 4)],
                             cv::Scalar(0, 255, 255), 1);
                }
                cv::circle(image, blade.center, 3, cv::Scalar(0, 0, 255), -1);
            }
            cv::drawMarker(image, rune.r_center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 1);
        }

        void drawPrediction(cv::Mat& image, const RuneSolver& solver, const RuneTarget& target,
                            const cv::Scalar& color, double roll)
        {
            const Eigen::VectorXd& state = target.ekfX();
            if (state.size() < 6) return;
            const Eigen::Vector3d center_in_world = target.pointBuffToWorld(Eigen::Vector3d::Zero());
            const auto corners = solver.reproject(center_in_world, state[4], roll);
            for (std::size_t i = 0; i + 1 < corners.size(); ++i) {
                cv::line(image, corners[i], corners[i + 1], color, 1);
            }
            if (corners.size() > 4) {
                cv::line(image, corners[4], corners[3], color, 1);
            }
        }
    } // namespace

    int presentFrame(const cv::Mat& frame, const RuneOverlayInput& overlay, bool show_display,
                     const char* dump_setting, bool dump_enabled)
    {
        if (!show_display && !dump_enabled) {
            // 没有窗口也没有落盘：让出 CPU（原实现如此，保持帧节奏一致）。
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return 1;
        }

        cv::Mat display = frame.clone();
        const RuneTarget& target = overlay.target;
        const RuneCommand& command = overlay.command;
        if (overlay.observation != nullptr) drawObservation(display, *overlay.observation);
        if (!target.isUnsolvable()) {
            drawPrediction(display, overlay.solver, target, cv::Scalar(0, 255, 0),
                           target.ekfX()[5]);
            drawPrediction(display, overlay.solver, overlay.predicted, cv::Scalar(255, 0, 255),
                           command.aim_roll);
            const Eigen::VectorXd& state = target.ekfX();
            cv::putText(display,
                        cv::format("mode=%s dis=%.2fm roll=%.1fdeg spd=%.1fdeg/s",
                                   overlay.mode == RuneMode::Large ? "large" : "small",
                                   state[3], state[5] * 180.0 / CV_PI,
                                   state[6] * 180.0 / CV_PI),
                        cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1,
                        cv::LINE_AA);
            cv::putText(display,
                        cv::format("aim yaw=%.1f pitch=%.1f deg t=%.3fs",
                                   command.yaw * 180.0 / CV_PI, command.pitch * 180.0 / CV_PI,
                                   command.fly_time),
                        cv::Point(8, 40), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 255),
                        1, cv::LINE_AA);
        }
        cv::putText(display,
                    cv::format("%s nn=%.1fms %s",
                               runeStatusName(overlay.detector.status()),
                               overlay.detector.latencyMs(),
                               command.shoot ? (overlay.fire_enabled ? "FIRE" : "FIRE(dry)")
                                             : (command.control ? "TRACK" : "SWITCH")),
                    cv::Point(8, 60), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1,
                    cv::LINE_AA);
        if (dump_enabled) {
            static int dumped = 0;
            const int every = std::max(1, std::atoi(dump_setting));
            if (dumped++ % every == 0) {
                cv::imwrite("/tmp/rune_dump_" + std::to_string(dumped) + ".png", display);
            }
        }
        if (show_display) {
            cv::imshow("Energy Rune", display);
            return cv::waitKey(1);
        }
        return 1;
    }

    const char* runeStatusName(RuneTrackStatus status)
    {
        switch (status) {
        case RuneTrackStatus::Track: return "TRACK";
        case RuneTrackStatus::TemporaryLost: return "TEMP_LOST";
        case RuneTrackStatus::Lost: return "LOST";
        }
        return "UNKNOWN";
    }

    RuneDiagRow makeRow(const RuneDiagSources& sources)
    {
        const RuneDetector& detector = sources.detector;
        const RuneTarget& target = sources.target;
        const PowerRune* rune = sources.rune;
        const RuneCommand& command = sources.command;

        RuneDiagRow row;
        row.local_time_us = sources.local_time_us;
        row.status = runeStatusName(detector.status());
        row.latency_ms = detector.latencyMs();
        row.has_rune = rune != nullptr;
        if (rune != nullptr) {
            row.r_center = rune->r_center;
            row.blade_center = rune->target().center;
        }

        const Eigen::VectorXd& state = target.ekfX();
        row.state_valid = !target.isUnsolvable() && state.size() >= 7;
        if (row.state_valid) {
            row.state_yaw = state[0];
            row.state_pitch = state[2];
            row.state_dis = state[3];
            row.measured_dis =
                (rune != nullptr && rune->solved) ? rune->ypd_in_world[2] : 0.0;
            row.state_center_yaw = state[4];
            row.state_roll = state[5];
            row.state_spd = state[6];
        }
        row.cmd_yaw = command.yaw;
        row.cmd_pitch = command.pitch;
        row.fly_time = command.fly_time;
        row.control = command.control ? 1 : 0;
        row.shoot = command.shoot ? 1 : 0;
        row.switched = command.blade_switched ? 1 : 0;
        row.target_yaw = command.yaw;
        row.target_pitch = command.pitch;
        row.sent_yaw = sources.sent_yaw;
        row.sent_pitch = sources.sent_pitch;
        row.class0 = detector.classCounts()[0];
        row.class1 = detector.classCounts()[1];
        row.class2 = detector.classCounts()[2];
        row.blade_class = detector.targetClass();
        // 观测片相对跟踪槽位的偏移（-1 = 该帧没有做折算，观测片就是跟踪槽位）。
        row.obs_slot_offset = (rune != nullptr && rune->slot_offset_valid)
                                  ? rune->slot_offset
                                  : -1;
        row.aim_phase_deg = sources.aim_phase_deg;
        row.ekf_roll_deg = state.size() > 5 ? state[5] * 180.0 / CV_PI : 0.0;
        // 靶面在图像里的等效半径（4 个角点到靶心的平均距离），用于判断远近/尺度。
        double plate_radius_px = 0.0;
        if (rune != nullptr && rune->target().points.size() >= 4) {
            const auto& points = rune->target().points;
            const cv::Point2f center = rune->target().center;
            for (int i = 0; i < 4; ++i) plate_radius_px += cv::norm(points[i] - center);
            plate_radius_px /= 4.0;
        }
        row.plate_r_px = plate_radius_px;
        row.cand_n = detector.candidateCount();
        row.cand_bright = detector.bestBladeBrightness();
        row.obs_phase_deg =
            (rune != nullptr && rune->phase_valid) ? rune->phase_rad * 180.0 / CV_PI : 0.0;
        row.pnp_roll_deg =
            rune != nullptr ? rune->ypr_in_world[2] * 180.0 / CV_PI : 0.0;
        row.sim_t = sources.sim_offset_valid
                        ? static_cast<double>(sources.local_time_us) / 1e6 - sources.sim_offset_s
                        : 0.0;
        row.best_cls_score = detector.modelBestScore();
        row.buff_yaw_deg = state.size() > 4 ? state[4] * 180.0 / CV_PI : 0.0;
        row.slot_id = sources.slot_id;
        row.slot_activated = sources.slot_activated;
        row.slot_votes = sources.slot_votes;
        row.slot_centers = sources.slot_centers;
        row.orbit_ok = sources.orbit_valid ? 1 : 0;
        row.orbit_samples = sources.orbit_samples;
        row.orbit_ratio = sources.orbit_ratio;
        row.orbit_semi_major_px = sources.orbit_semi_major_px;
        row.orbit_center = sources.orbit_center;
        row.sigma_dis_m = sources.sigma_dis_m;
        row.sigma_roll_deg = sources.sigma_roll_deg;
        row.slot_scores = sources.slot_scores;
        row.slot_present_mask = sources.slot_present_mask;
        row.slot_found_count = sources.slot_found_count;
        row.slot_found_centers = sources.slot_found_centers;
        // 靶面压扁程度：正方形靶面的两条对角线在正对时相等，侧视时其中一条被压缩。
        if (rune != nullptr && rune->target().points.size() >= 4) {
            const auto& points = rune->target().points;
            const double d0 = cv::norm(points[0] - points[2]);
            const double d1 = cv::norm(points[1] - points[3]);
            const double hi = std::max(d0, d1);
            if (hi > 1e-6) row.plate_aspect = std::min(d0, d1) / hi;
        }
        row.aim_pixel = sources.aim_pixel;
        row.aim_world_yaw_deg =
            std::atan2(sources.aim_world[1], sources.aim_world[0]) * 180.0 / CV_PI;
        row.aim_world_pitch_deg =
            std::atan2(sources.aim_world[2],
                       std::hypot(sources.aim_world[0], sources.aim_world[1])) *
            180.0 / CV_PI;
        row.aim_roll_deg = sources.aim_roll_deg;
        row.refine_n = detector.refineStats().candidates;
        row.refine_usable = detector.refineStats().usable;
        row.refine_dropped = detector.refineStats().dropped;
        row.refine_fused = detector.refineStats().center_fused;
        row.state_lit_ratio = detector.refineStats().state_lit_ratio;
        row.state_arm_ratio = detector.refineStats().state_arm_ratio;
        row.state_code = detector.refineStats().state_code;
        row.state_orbit_px = detector.refineStats().state_orbit_px;
        row.state_lit_area = detector.refineStats().state_lit_area;
        row.state_roi_px = detector.refineStats().state_roi_px;
        row.classic_lit_area = detector.refineStats().classic_lit_area;
        row.classic_lit_ratio = detector.refineStats().classic_lit_ratio;
        row.plate_refined = detector.refineStats().plate_refined;
        row.plate_size_ratio = detector.refineStats().last_plate_size_ratio;
        row.armor_solidity = detector.refineStats().last_armor_solidity;
        row.armor_area_ratio = detector.refineStats().last_armor_area_ratio;
        row.armor_circularity = detector.refineStats().last_armor_circularity;
        row.refine_arm_solidity = detector.refineStats().last_arm_solidity;
        row.refine_arm_aspect = detector.refineStats().last_arm_aspect;
        row.refine_gap_px = detector.refineStats().last_gap_px;
        row.k2_px = detector.lastBladeRMark();
        row.hub_src = static_cast<int>(detector.hubSource());
        row.hub_plate_dist_px =
            cv::norm(detector.lastBladeRMark() - detector.lastBladeCenter());
        row.target_valid = target.isUnsolvable() ? 0 : 1;
        row.blade_switch_obs = target.bladeSwitched() ? 1 : 0;
        row.picked_bright = detector.pickedBladeBrightness();
        row.assoc_reject = detector.phaseAssocRejected() ? 1 : 0;
        row.assoc_residual_deg = detector.phaseAssocResidualRad() >= 0.0
                                     ? detector.phaseAssocResidualRad() * 180.0 / CV_PI
                                     : -1.0;
        row.lit_px = detector.lastLitBladeCenter();
        row.picked_px = detector.lastBladeCenter();
        row.fired = sources.fired ? 1 : 0;
        row.gate_hold = sources.gate_hold ? 1 : 0;
        row.gate_window_open = sources.gate_window_open ? 1 : 0;
        row.gate_engageable = sources.gate_engageable ? 1 : 0;
        row.aim_err_yaw_deg = sources.aim_err_yaw_deg;
        row.aim_err_pitch_deg = sources.aim_err_pitch_deg;
        row.diff_ok = sources.diff.ok;
        row.diff_px = sources.diff.px;
        row.diff_py = sources.diff.py;
        row.diff_area = sources.diff.area;
        row.diff_to_lit_px = sources.diff.to_lit_px;
        for (const auto& candidate : detector.lastCandidates()) {
            row.candidates += std::to_string(candidate.class_id) + ':' +
                              std::to_string(static_cast<int>(candidate.center.x)) + ':' +
                              std::to_string(static_cast<int>(candidate.center.y)) + '|';
        }
        return row;
    }

    bool openRecorder(std::ofstream& recorder, const std::string& path)
    {
        recorder.open(path, std::ios::out | std::ios::trunc);
        if (!recorder) return false;
        recorder << "time_us,status,latency_ms,r_center_x,r_center_y,cx,cy,R_yaw,R_pitch,"
                    "R_dis,meas_dis,yaw,roll,spd,aim_yaw,aim_pitch,fly_time,control,shoot,"
                    "switched,target_yaw,target_pitch,sent_yaw,sent_pitch,"
                    "class0,class1,class2,blade_class,"
                    "aim_phase_deg,ekf_roll_deg,plate_r_px,cand_n,cand_bright,"
                    "obs_phase_deg,pnp_roll_deg,sim_t,best_cls_score,buff_yaw_deg,slot_id,"
                    "slot_activated_mask,slot_votes,slot_centers,obs_slot_offset,orbit_ok,"
                    "orbit_a_px,orbit_cx,orbit_cy,orbit_n,orbit_ratio,sigma_dis_m,sigma_roll_deg,"
                    "slot_present_mask,slot_scores,slot_found_n,slot_found,plate_aspect,aim_px,aim_py,"
                    "aim_world_yaw_deg,"
                    "aim_world_pitch_deg,"
                    "aim_roll_deg,refine_n,refine_usable,refine_dropped,refine_fused,"
                    "state_code,state_lit_ratio,state_arm_ratio,classic_lit_area,classic_lit_ratio,plate_refined,plate_size_ratio,armor_solidity,armor_area_ratio,armor_circularity,"
                    "refine_arm_solidity,refine_arm_aspect,refine_gap_px,k2_px,k2_py,hub_src,"
                    "hub_plate_dist_px,target_valid,blade_switch_obs,picked_bright,"
                    "assoc_reject,assoc_residual_deg,lit_px,lit_py,picked_px,picked_py,fired,"
                    "aim_err_yaw_deg,aim_err_pitch_deg,diff_ok,diff_px,diff_py,diff_area,"
                    "diff_to_lit_px,cands,state_orbit_px,state_lit_area,state_roi_px,"
                    "gate_hold,gate_window_open,gate_engageable\n";
        return true;
    }

    void writeRow(std::ofstream& recorder, const RuneDiagRow& row)
    {
        if (!recorder) return;
        recorder << std::fixed << std::setprecision(6) << row.local_time_us << ',' << row.status
                 << ',' << row.latency_ms << ',';
        if (row.has_rune) {
            recorder << row.r_center.x << ',' << row.r_center.y << ',' << row.blade_center.x
                     << ',' << row.blade_center.y << ',';
        } else {
            recorder << "0,0,0,0,";
        }
        if (row.state_valid) {
            recorder << row.state_yaw << ',' << row.state_pitch << ',' << row.state_dis << ','
                     << row.measured_dis << ',' << row.state_center_yaw << ',' << row.state_roll
                     << ',' << row.state_spd << ',';
        } else {
            recorder << "0,0,0,0,0,0,0,";
        }
        recorder << row.cmd_yaw << ',' << row.cmd_pitch << ',' << row.fly_time << ','
                 << row.control << ',' << row.shoot << ',' << row.switched << ','
                 << row.target_yaw << ',' << row.target_pitch << ',' << row.sent_yaw << ','
                 << row.sent_pitch << ',' << row.class0 << ',' << row.class1 << ','
                 << row.class2 << ',' << row.blade_class << ',' << row.aim_phase_deg << ','
                 << row.ekf_roll_deg << ',' << row.plate_r_px << ',' << row.cand_n << ','
                 << row.cand_bright << ',' << row.obs_phase_deg << ',' << row.pnp_roll_deg << ','
                 << row.sim_t << ',' << row.best_cls_score << ',' << row.buff_yaw_deg << ','
                 << row.slot_id << ',' << (row.slot_activated[0] ? 1 : 0)
                 << (row.slot_activated[1] ? 1 : 0) << (row.slot_activated[2] ? 1 : 0)
                 << (row.slot_activated[3] ? 1 : 0) << (row.slot_activated[4] ? 1 : 0) << ','
                 << row.slot_votes[0] << '/' << row.slot_votes[1] << '/' << row.slot_votes[2]
                 << '/' << row.slot_votes[3] << '/' << row.slot_votes[4] << ','
                 << static_cast<int>(row.slot_centers[0].x) << ':'
                 << static_cast<int>(row.slot_centers[0].y) << '|'
                 << static_cast<int>(row.slot_centers[1].x) << ':'
                 << static_cast<int>(row.slot_centers[1].y) << '|'
                 << static_cast<int>(row.slot_centers[2].x) << ':'
                 << static_cast<int>(row.slot_centers[2].y) << '|'
                 << static_cast<int>(row.slot_centers[3].x) << ':'
                 << static_cast<int>(row.slot_centers[3].y) << '|'
                 << static_cast<int>(row.slot_centers[4].x) << ':'
                 << static_cast<int>(row.slot_centers[4].y) << ',' << row.obs_slot_offset << ','
                 << row.orbit_ok << ',' << row.orbit_semi_major_px << ',' << row.orbit_center.x
                 << ',' << row.orbit_center.y << ',' << row.orbit_samples << ','
                 << row.orbit_ratio << ',' << row.sigma_dis_m << ',' << row.sigma_roll_deg << ',' << row.slot_present_mask << ',' << row.slot_scores << ',' << row.slot_found_count << ','
                 << static_cast<int>(row.slot_found_centers[0].x) << ':' << static_cast<int>(row.slot_found_centers[0].y) << '|'
                 << static_cast<int>(row.slot_found_centers[1].x) << ':' << static_cast<int>(row.slot_found_centers[1].y) << '|'
                 << static_cast<int>(row.slot_found_centers[2].x) << ':' << static_cast<int>(row.slot_found_centers[2].y) << '|'
                 << static_cast<int>(row.slot_found_centers[3].x) << ':' << static_cast<int>(row.slot_found_centers[3].y) << '|'
                 << static_cast<int>(row.slot_found_centers[4].x) << ':' << static_cast<int>(row.slot_found_centers[4].y) << ','
                 << row.plate_aspect << ','
                 << row.aim_pixel.x << ',' << row.aim_pixel.y << ',' << row.aim_world_yaw_deg
                 << ',' << row.aim_world_pitch_deg << ',' << row.aim_roll_deg << ','
                 << row.refine_n << ',' << row.refine_usable << ',' << row.refine_dropped << ','
                 << row.refine_fused << ',' << row.state_code << ',' << row.state_lit_ratio << ',' << row.state_arm_ratio << ',' << row.classic_lit_area << ',' << row.classic_lit_ratio << ',' << row.plate_refined << ',' << row.plate_size_ratio << ',' << row.armor_solidity << ',' << row.armor_area_ratio
                 << ',' << row.armor_circularity << ',' << row.refine_arm_solidity << ','
                 << row.refine_arm_aspect << ',' << row.refine_gap_px << ',' << row.k2_px.x << ','
                 << row.k2_px.y << ',' << row.hub_src << ',' << row.hub_plate_dist_px << ','
                 << row.target_valid << ',' << row.blade_switch_obs << ',' << row.picked_bright
                 << ',' << row.assoc_reject << ',' << row.assoc_residual_deg << ','
                 << row.lit_px.x << ',' << row.lit_px.y << ',' << row.picked_px.x << ','
                 << row.picked_px.y << ',' << row.fired << ',' << row.aim_err_yaw_deg << ','
                 << row.aim_err_pitch_deg << ',' << row.diff_ok << ',' << row.diff_px << ','
                 << row.diff_py << ',' << row.diff_area << ',' << row.diff_to_lit_px << ','
                 << row.candidates << ',' << row.state_orbit_px << ',' << row.state_lit_area
                 << ',' << row.state_roi_px << ',' << row.gate_hold << ','
                 << row.gate_window_open << ',' << row.gate_engageable << '\n';
    }


    DiffSample probeDiff(RuneDiffProbe& probe, const cv::Mat& frame, cv::Point2f hub_px,
                         double orbit_px, cv::Point2f lit_px)
    {
        DiffSample sample;
        if (const auto candidate = probe.update(frame, hub_px, orbit_px)) {
            sample.ok = 1.0;
            sample.px = candidate->center.x;
            sample.py = candidate->center.y;
            sample.area = candidate->area;
            if (lit_px.x > 0.0f) sample.to_lit_px = cv::norm(candidate->center - lit_px);
        }
        return sample;
    }

    void consumeTelemetry(const std::vector<std::string>& lines, std::ofstream* ground_truth,
                          const std::string& rune_mode_name, RuneTelemetry& telemetry,
                          bool* scored_hit_this_frame)
    {
        for (const std::string& line : lines) {
            const std::string mode = telemetryField(line, "mode");
            if (mode != rune_mode_name) continue;

            // 遥测里带仿真时钟 t=：用它算"本地时钟 - 仿真时钟"的偏移，让 CSV 能写出
            // 仿真时间，从而把视觉曲线和仿真里的轮次/命中对齐。
            // 注意：不能按 "t=" 找子串——"event=" 里也有 "t="，必须匹配独立字段。
            {
                const auto position = line.find(" t=");
                const double sim_seconds =
                    position == std::string::npos ? 0.0
                                                  : std::atof(line.c_str() + position + 3);
                if (sim_seconds > 0.0) {
                    telemetry.sim_offset_s = StandardClock::nowSeconds() - sim_seconds;
                    telemetry.sim_offset_valid = true;
                }
            }

            const std::string event = telemetryField(line, "event");
            const std::string phase = telemetryField(line, "phase");
            if (event == "hit") {
                if (telemetryField(line, "accurate") == "1") ++telemetry.hits;
                if (telemetryField(line, "scored") == "1") {
                    ++telemetry.scored;
                    if (scored_hit_this_frame != nullptr) *scored_hit_this_frame = true;
                }
                std::cout << "  gt " << line << std::endl;
            }
            if (event == "activated" || telemetryField(line, "change") == "1") {
                ++telemetry.activations;
                if (telemetry.first_activation_s < 0.0) {
                    telemetry.first_activation_s = std::atof(telemetryField(line, "t").c_str());
                }
                std::cout << "  gt " << line << std::endl;
            }
            if (phase == "failed" && event == "state") {
                std::cout << "  gt rune failed, retrying" << std::endl;
            }
            if (ground_truth) {
                *ground_truth << StandardClock::nowUs() << ',' << line << '\n';
            }
        }
    }

    void dumpFrame(const std::string& frame_dir, const cv::Mat& frame, int frame_index, int& saved)
    {
        if (frame_dir.empty() || frame.empty()) return;
        static int stride = -1;
        if (stride < 0) {
            const char* value = std::getenv("ULTRA_VISION_RUNE_FRAME_STRIDE");
            stride = value ? std::atoi(value) : 0;   // 0 = 不录制
        }
        if (stride <= 0 || frame_index % stride != 0) return;

        char name[64];
        std::snprintf(name, sizeof(name), "/frame_%06d.jpg", ++saved);
        cv::imwrite(frame_dir + name, frame);
    }

    void printSummary(int shots_fired, const RuneTelemetry& telemetry, int shots_held_for_window,
                      int parked_frames, int recenter_count,
                      const std::vector<double>& gimbal_steps_deg,
                      const std::vector<double>& gimbal_rates_deg_s)
    {
        std::cout << "=== rune run summary (simulator ground truth) ===" << std::endl;
        std::cout << "  shots fired     : " << shots_fired << std::endl;
        std::cout << "  hits on target  : " << telemetry.hits << std::endl;
        std::cout << "  scored hits     : " << telemetry.scored
                  << "  (blade was lit and not yet hit)" << std::endl;
        std::cout << "  activations     : " << telemetry.activations << std::endl;
        std::cout << "  shots held back : " << shots_held_for_window
                  << "  (would land after the round window closed)" << std::endl;
        std::cout << "  parked frames   : " << parked_frames
                  << "  (丢目标时停在符心，而不是回标定位姿)" << std::endl;
        std::cout << "  recenter count  : " << recenter_count << std::endl;
        if (!gimbal_steps_deg.empty()) {
            std::cout << "  gimbal commands : " << gimbal_steps_deg.size()
                      << "  step p50=" << percentile(gimbal_steps_deg, 0.5)
                      << " p90=" << percentile(gimbal_steps_deg, 0.9)
                      << " max=" << percentile(gimbal_steps_deg, 1.0) << " deg"
                      << "  rate p90=" << percentile(gimbal_rates_deg_s, 0.9)
                      << " max=" << percentile(gimbal_rates_deg_s, 1.0) << " deg/s" << std::endl;
        }
        if (telemetry.first_activation_s >= 0.0) {
            std::cout << "  activation at   : " << telemetry.first_activation_s
                      << " s (sim clock)" << std::endl;
        }
    }

    BoreSight BoreSight::fromEnvironment()
    {
        BoreSight bore;
        if (const char* env = std::getenv("ULTRA_VISION_RUNE_BORESIGHT")) {
            double yaw_deg = 0.0;
            double pitch_deg = 0.0;
            if (std::sscanf(env, "%lf,%lf", &yaw_deg, &pitch_deg) == 2) {
                bore.enabled = true;
                bore.yaw = yaw_deg * CV_PI / 180.0;
                bore.pitch = pitch_deg * CV_PI / 180.0;
                std::cout << "Energy rune: BORE-SIGHT 模式 yaw=" << yaw_deg
                          << " deg pitch=" << pitch_deg << " deg" << std::endl;
            }
        }
        if (const char* env = std::getenv("ULTRA_VISION_RUNE_BORESIGHT_INTERVAL")) {
            bore.interval_s = std::atof(env);
            if (bore.interval_s < 0.05) bore.interval_s = 0.05;
        }
        return bore;
    }

    void BoreSight::apply(RuneCommand& command, double now)
    {
        if (!enabled) return;
        command.control = true;
        command.yaw = yaw;
        command.pitch = pitch;
        command.yaw_velocity = 0.0;
        command.pitch_velocity = 0.0;
        command.blade_switched = false;
        command.shoot = (now - last_shot_time) >= interval_s;
        if (command.shoot) last_shot_time = now;
    }
} // namespace auto_aim::energy::rune_diag
