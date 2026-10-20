#include "auto_buff/rune_detector.hpp"
#include "auto_buff/support/math.hpp"

#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include "common/standard_clock.hpp"

namespace auto_aim::energy
{
    RuneDetector::RuneDetector(const RuneDetectorConfig& config)
        : config_(config), model_(config.model), refiner_(config.refiner),
          plate_refiner_(config.plate_refiner)
    {
        if (const char* adaptive = std::getenv("ULTRA_VISION_RUNE_ADAPTIVE_SCALE")) {
            config_.canvas_adaptive = std::string(adaptive) != "0";
        }
        if (const char* recenter = std::getenv("ULTRA_VISION_RUNE_ROI_CENTER")) {
            config_.canvas_recenter = std::string(recenter) != "0";
        }
        if (const char* fallback = std::getenv("ULTRA_VISION_RUNE_ROI_FALLBACK")) {
            config_.canvas_recenter_fallback = std::string(fallback) != "0";
        }
        lifecycle_.configure(config_.status_track_hold_s,
                             config_.status_lost_timeout_s);
        // 启动横幅：把"这次到底用了哪套模型/阈值"打出来 —— 现场排查最常见的
        std::cout << "[rune config] classes=" << config_.model.num_classes
                  << " keypoints=" << config_.model.num_keypoints
                  << " input=" << config_.model.input_size << "x"
                  << (config_.model.input_height > 0 ? config_.model.input_height
                                                    : config_.model.input_size)
                  << " pad_scale=" << config_.model.input_pad_scale
                  << " input_scale=" << config_.model.input_scale
                  << " score_thr=" << config_.model.confidence_threshold
                  << " kpt_conf=" << config_.model.keypoint_confidence_threshold
                  << " min_kpt=" << config_.model.min_valid_keypoints
                  << " pad_adaptive=" << (config_.canvas_adaptive ? 1 : 0)
                  << " pad_recenter=" << (config_.canvas_recenter ? 1 : 0)
                  << " roi_fallback=" << (config_.canvas_recenter_fallback ? 1 : 0)
                  << " pad_target_orbit_px=" << config_.canvas_target_orbit_px
                  << " model=" << config_.model.model_path << std::endl;
    }

    void RuneDetector::handleLost(double timestamp)
    {
        lifecycle_.noteRejected(timestamp);
        last_eval_time_ = timestamp;
        if (lifecycle_.currentStatus() == RuneTrackStatus::Lost) {
            last_powerrune_ = std::nullopt;
        }
    }

    cv::Point2f RuneDetector::estimateRuneCenter(std::vector<FanBlade>& blades,
                                                const cv::Mat& bgr_image,
                                                const std::optional<cv::Point2f>& multi_blade_hub)
    {
        if (blades.empty()) return {0.0f, 0.0f};

        // 0) 多片扇叶求交得到的圆心最可信（每片都给出一条"圆心→靶心"射线，
        if (multi_blade_hub.has_value()) {
            hub_source_ = HubSource::MultiBladeRay;
            return multi_blade_hub.value();
        }

        // 靶面点半径（像素）：四个靶面点相对靶心的平均距离。圆心一定在
        double plate_radius_px = 0.0;
        {
            const std::size_t count = std::min<std::size_t>(4, blades.front().points.size());
            for (std::size_t i = 0; i < count; ++i) {
                plate_radius_px += cv::norm(blades.front().points[i] - blades.front().center);
            }
            plate_radius_px = count > 0 ? plate_radius_px / static_cast<double>(count) : 0.0;
        }
        plate_radius_px = std::max(2.0, plate_radius_px);

        // 1) 网络直接给的 R 标（旋转中心）。深大五点模型的第 5 个点就是它，
        cv::Point2f hint(0.0f, 0.0f);
        int hint_count = 0;
        const double orbit_ratio = std::max(1.5, config_.center_orbit_ratio);
        for (const FanBlade& blade : blades) {
            if (!blade.has_rune_center) continue;
            const double offset = cv::norm(blade.rune_center - blade.center);
            // 圆心到靶心的像素距离应当 ≈ orbit_ratio × 靶面点半径：太近说明网络把
            if (offset < 0.5 * plate_radius_px ||
                offset > std::max(2.0, 1.5 * orbit_ratio) * plate_radius_px) {
                if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                    std::cerr << "[rune center] R-mark hint rejected: offset " << offset
                              << " px vs plate_r " << plate_radius_px << std::endl;
                }
                continue;
            }
            hint += blade.rune_center;
            ++hint_count;
        }
        const bool have_hint = hint_count > 0;
        if (have_hint) hint /= static_cast<float>(hint_count);

        // 模型给的 R 标（旋转中心）是专门训练出来的点，深大参考实现也直接
        if (have_hint) {
            hub_source_ = HubSource::RMarkKeypoint;
            return hint;
        }

        // 3) 没有可用方向：沿环带找"最亮方位"。能量机关的 R 标是画面里唯一
        {
            const cv::Point2f plate_center = blades.front().center;
            // 判据是"小亮点 + 周围暗"：R 标是几像素大小的亮块（仿真是纯红，
            cv::Mat channels[3];
            cv::split(bgr_image, channels);
            const cv::Mat bright = cv::max(cv::max(channels[0], channels[1]), channels[2]);
            const auto patch_max = [&](const cv::Point2f& point, int half) {
                cv::Rect roi(static_cast<int>(std::lround(point.x)) - half,
                             static_cast<int>(std::lround(point.y)) - half, 2 * half + 1,
                             2 * half + 1);
                roi &= cv::Rect(0, 0, bright.cols, bright.rows);
                if (roi.width < 2 || roi.height < 2) return -1.0;
                double minimum = 0.0;
                double maximum = 0.0;
                cv::minMaxLoc(bright(roi), &minimum, &maximum);
                return maximum;
            };
            const auto patch_mean = [&](const cv::Point2f& point, int half) {
                cv::Rect roi(static_cast<int>(std::lround(point.x)) - half,
                             static_cast<int>(std::lround(point.y)) - half, 2 * half + 1,
                             2 * half + 1);
                roi &= cv::Rect(0, 0, bright.cols, bright.rows);
                if (roi.width < 2 || roi.height < 2) return 255.0;
                return cv::mean(bright(roi))[0];
            };

            const int half_small = 2;
            const int half_large =
                std::max(4, static_cast<int>(std::lround(plate_radius_px * 0.8)));
            double best_score = -1e9;
            double expected_distance = 0.0;
            cv::Point2f best = plate_center;
            for (double ratio = 1.5; ratio <= 8.001; ratio += 0.25) {
                const double distance_px = ratio * plate_radius_px;
                for (int step = 0; step < 360; step += 3) {
                    const double angle = step * CV_PI / 180.0;
                    const cv::Point2f candidate(
                        plate_center.x + static_cast<float>(distance_px * std::cos(angle)),
                        plate_center.y + static_cast<float>(distance_px * std::sin(angle)));
                    const double score =
                        patch_max(candidate, half_small) - patch_mean(candidate, half_large);
                    if (score > best_score) {
                        best_score = score;
                        best = candidate;
                        expected_distance = distance_px;
                    }
                }
            }

            const bool ring_confident = best_score > config_.hub_brightness_margin;
            if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                std::cerr << "[rune center] plate_r=" << plate_radius_px
                          << " ring=(" << best.x << "," << best.y << ") r=" << expected_distance
                          << " score=" << best_score << " confident=" << ring_confident
                          << std::endl;
            }
            hub_source_ = HubSource::RingBrightness;
            if (ring_confident) return best;
            return plate_center;
        }
    }

    std::optional<PowerRune> RuneDetector::buildRune(std::vector<RuneModel::Object>&& results,
                                                     const cv::Mat& bgr_image, double timestamp)
    {
        // 类别统计（CSV 与激活状态机用）：0=未激活，1=小符已激活，2=大符已激活。
        class_counts_ = {0, 0, 0};
        // 诊断量每帧清零：否则没建出 rune 的帧会沿用上一帧的值，
        last_blade_center_ = {0.0f, 0.0f};
        last_blade_rmark_ = {0.0f, 0.0f};
        frame_orbit_px_ = -1.0;
        lit_center_ = {0.0f, 0.0f};
        target_class_ = -1;
        picked_by_geometry_ = false;
        candidate_count_ = static_cast<int>(results.size());
        best_brightness_ = 0.0;
        // 过滤前的候选（含已激活片）留一份，供上层做"槽位已激活"的语义记账。
        last_candidates_.clear();
        last_candidates_.reserve(results.size());
        for (const RuneModel::Object& candidate : results) {
            CandidateInfo info;
            info.class_id = candidate.class_id;
            info.center = candidate.center;
            // 五点模型：keypoints[0,1,3,4] 是靶面四点（边中点），[2] 是 R 标。
            if (candidate.keypoints.size() >= 5) {
                info.plate_points = {candidate.keypoints[0], candidate.keypoints[1],
                                     candidate.keypoints[3], candidate.keypoints[4]};
                info.rmark = candidate.keypoints[2];
                info.has_rmark = true;
                info.has_plate = true;
            }
            last_candidates_.push_back(info);
        }
        for (const RuneModel::Object& candidate : results) {
            if (candidate.class_id >= 0 && candidate.class_id < 3) {
                ++class_counts_[static_cast<std::size_t>(candidate.class_id)];
            }
        }
        const auto plateCenter = [](const RuneModel::Object& candidate) {
            return candidate.center;
        };

        // ① 几何修正：每片扇叶（含已激活）都给出"圆心→靶心"方向，圆心必在这些
        std::optional<cv::Point2f> multi_blade_hub;
        {
            Eigen::Matrix2d normal_matrix = Eigen::Matrix2d::Zero();
            Eigen::Vector2d rhs = Eigen::Vector2d::Zero();
            std::vector<std::pair<cv::Point2f, Eigen::Vector2d>> rays;
            for (const RuneModel::Object& candidate : results) {
                if (candidate.keypoints.size() < 5) continue;
                const cv::Point2f r_mark = candidate.keypoints[2];   // R 标 ≈ 圆心
                const cv::Point2f plate = candidate.center;          // 靶心
                const cv::Point2f delta = plate - r_mark;
                const float length = cv::norm(delta);
                if (length < 1.0f) continue;
                const Eigen::Vector2d direction(-delta.x / length, -delta.y / length);
                const Eigen::Vector2d normal(-direction.y(), direction.x());
                normal_matrix += normal * normal.transpose();
                rhs += normal * normal.dot(Eigen::Vector2d(plate.x, plate.y));
                rays.emplace_back(plate, normal);
            }
            if (rays.size() >= 2 && std::abs(normal_matrix.determinant()) > 1e-6) {
                const Eigen::Vector2d solved = normal_matrix.ldlt().solve(rhs);
                const cv::Point2f hub(static_cast<float>(solved.x()),
                                      static_cast<float>(solved.y()));
                bool consistent = hub.x > -0.25f * bgr_image.cols &&
                                  hub.x < 1.25f * bgr_image.cols &&
                                  hub.y > -0.25f * bgr_image.rows &&
                                  hub.y < 1.25f * bgr_image.rows;
                for (const auto& ray : rays) {
                    const Eigen::Vector2d offset(ray.first.x - hub.x, ray.first.y - hub.y);
                    const double reference = offset.norm();
                    if (reference < 1.0) { consistent = false; break; }
                    if (std::abs(offset.dot(ray.second)) > 0.35 * reference) {
                        consistent = false;
                        break;
                    }
                }
                if (consistent) multi_blade_hub = hub;
            }
            if (std::getenv("ULTRA_VISION_RUNE_DEBUG") && multi_blade_hub.has_value()) {
                std::cerr << "[rune hub] multi-blade intersection: (" << multi_blade_hub->x << ","
                          << multi_blade_hub->y << ") from " << rays.size() << " blades"
                          << std::endl;
            }
        }

        // ② 目标只取"未激活(class 0)"：小符每轮只有一片未激活，天然不会在两片
        const auto candidate_phase = [&](const RuneModel::Object& candidate) -> double {
            if (candidate.keypoints.size() < 5) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const cv::Point2f hub = candidate.keypoints[2];
            const cv::Point2f plate = plateCenter(candidate);
            return std::atan2(plate.y - hub.y, plate.x - hub.x);
        };
        // ---- 几何连续优先（必须在类别过滤**之前**）-----------------------------
        std::optional<std::size_t> continuous_index;
        if (config_.phase_assoc.enabled && assoc_valid_ && timestamp > 0.0) {
            const double dt = timestamp - assoc_time_;
            if (dt > 1e-4 && dt <= config_.phase_assoc.max_gap_s) {
                const double predicted = assoc_phase_ + assoc_rate_ * dt;
                const double gate = std::min(
                    0.55 * (2.0 * CV_PI / 5.0),
                    config_.phase_assoc.tolerance_rad + 0.5 * dt);
                double best_cost = std::numeric_limits<double>::max();
                for (std::size_t i = 0; i < results.size(); ++i) {
                    const double phase = candidate_phase(results[i]);
                    if (!std::isfinite(phase)) continue;
                    const double cost = std::abs(limitRad(phase - predicted));
                    if (cost < best_cost) {
                        best_cost = cost;
                        continuous_index = i;
                    }
                }
                if (continuous_index.has_value() && best_cost > gate) {
                    continuous_index.reset();   // 离预测太远：本帧没有"连续的那片"
                }
            }
        }
        if (continuous_index.has_value()) {
            std::vector<RuneModel::Object> keep;
            keep.push_back(results[*continuous_index]);
            results.swap(keep);
            picked_by_geometry_ = true;
            if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                std::cerr << "[rune pick] geometry-continuous candidate kept (class "
                          << results.front().class_id << "), observation is class-agnostic"
                          << std::endl;
            }
        }
        // Observation/model construction is class-agnostic. Activated blades are

        if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
            std::cerr << "[rune detect] candidates=" << results.size();
            for (const RuneModel::Object& candidate : results) {
                std::cerr << " | cls" << candidate.class_id << " p=" << candidate.prob
                          << " c=(" << candidate.center.x << "," << candidate.center.y << ")";
                for (const cv::Point2f& point : candidate.keypoints) {
                    std::cerr << " (" << point.x << "," << point.y << ")";
                }
            }
            std::cerr << std::endl;
        }

        // Keep only the candidate whose plate is actually lit. Several blades
        if (config_.prefer_bright_blade && !results.empty()) {
            // 亮点判据用"小窗峰值 − 大窗均值"，而不是区域均值：仿真里画面
            cv::Mat channels[3];
            cv::split(bgr_image, channels);
            const cv::Mat bright_image =
                cv::max(cv::max(channels[0], channels[1]), channels[2]);
            const auto brightness = [&](const RuneModel::Object& candidate) {
                const cv::Point2f center = plateCenter(candidate);
                const float half = 0.5f * static_cast<float>(config_.blade_roi_ratio) *
                                   std::max(candidate.rect.width, candidate.rect.height);
                cv::Rect roi(static_cast<int>(center.x - half), static_cast<int>(center.y - half),
                             std::max(2, static_cast<int>(2 * half)),
                             std::max(2, static_cast<int>(2 * half)));
                roi &= cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
                if (roi.width <= 0 || roi.height <= 0) return 0.0;
                // 小窗固定 5x5（靶面图案在 6 m 处只有 ~15 px 半径，用更大的窗
                const int small = 2;
                cv::Rect small_roi(static_cast<int>(center.x) - small,
                                   static_cast<int>(center.y) - small, 2 * small + 1,
                                   2 * small + 1);
                small_roi &= cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
                if (small_roi.width <= 0 || small_roi.height <= 0) return 0.0;
                double minimum = 0.0;
                double peak = 0.0;
                cv::minMaxLoc(bright_image(small_roi), &minimum, &peak);
                return peak - cv::mean(bright_image(roi))[0];
            };

            std::vector<double> scores(results.size());
            for (std::size_t i = 0; i < results.size(); ++i) scores[i] = brightness(results[i]);
            candidate_count_ = static_cast<int>(results.size());
            best_brightness_ = scores.empty()
                                   ? 0.0
                                   : *std::max_element(scores.begin(), scores.end());
            picked_brightness_ = 0.0;
            lit_center_ = {0.0f, 0.0f};
            if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                std::cerr << "[rune pick]";
                for (std::size_t i = 0; i < results.size(); ++i) {
                    std::cerr << " cls" << results[i].class_id << " p=" << results[i].prob
                              << " c=(" << results[i].center.x << "," << results[i].center.y
                              << ") bright=" << scores[i];
                }
                std::cerr << std::endl;
            }

            // 记录所有"足够亮"的候选作为点亮扇叶集合（阈值取最亮者的 60%，
            lit_centers_.clear();
            const double lit_threshold = *std::max_element(scores.begin(), scores.end()) * 0.6;
            for (std::size_t i = 0; i < results.size(); ++i) {
                if (scores[i] < lit_threshold) continue;
                lit_centers_.push_back(plateCenter(results[i]));
            }

            std::size_t best = 0;
            for (std::size_t i = 1; i < results.size(); ++i) {
                if (scores[i] > scores[best]) best = i;
            }
            // 最亮候选 = 点亮的那片（亮点判据排序的结果），记录它的位置。
            lit_center_ = plateCenter(results[best]);

            // ① 相位关联选片（优先于亮度竞争，且**不依赖 EKF**）
            bool assoc_used = false;
            assoc_rejected_ = false;
            std::size_t assoc_choice = best;
            double assoc_residual = 0.0;
            if (config_.phase_assoc.enabled && timestamp > 0.0) {
                const auto& assoc = config_.phase_assoc;
                assoc_residual_last_ = -1.0;
                const double dt = assoc_valid_ ? timestamp - assoc_time_ : -1.0;
                const bool can_predict = assoc_valid_ && dt > 1e-4 && dt <= assoc.max_gap_s;

                if (can_predict) {
                    // 关联判据（按预测相位取最近 + 残差门限）：
                    const double predicted = assoc_phase_ + assoc_rate_ * dt;
                    const double gate =
                        std::min(0.55 * (2.0 * CV_PI / 5.0), assoc.tolerance_rad + 0.5 * dt);
                    double best_cost = std::numeric_limits<double>::max();
                    std::size_t index = 0;
                    bool found = false;
                    bool direction_rejected = false;
                    for (std::size_t i = 0; i < results.size(); ++i) {
                        if (results[i].keypoints.size() < 5) continue;
                        const double delta = limitRad(candidate_phase(results[i]) - assoc_phase_);
                        const double implied = delta / dt;
                        // 方向必须一致（机关的旋转方向一次确定后不变）。
                        if (assoc_sign_ != 0 && (implied > 0) != (assoc_sign_ > 0)) {
                            direction_rejected = true;
                            continue;
                        }
                        const double cost = std::abs(limitRad(candidate_phase(results[i]) -
                                                              predicted));
                        if (cost < best_cost) {
                            best_cost = cost;
                            index = i;
                            found = true;
                        }
                    }
                    if (found && best_cost > gate) found = false;
                    // 连续多帧只因方向被拒 → 保存的方向可能过期，清掉重新捕获。
                    if (!found && direction_rejected && ++assoc_sign_mismatch_ >= 3) {
                        assoc_sign_ = 0;
                        assoc_sign_mismatch_ = 0;
                        if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                            std::cerr << "[rune pick] association direction reset (mismatch x3)"
                                      << std::endl;
                        }
                    } else if (found) {
                        assoc_sign_mismatch_ = 0;
                    }
                    if (found) {
                        assoc_choice = index;
                        assoc_used = true;
                        assoc_residual = best_cost;
                        pending_switch_frames_ = 0;
                    } else if (results.size() > 1) {
                        // 只有"多片同时可见"时才用关联丢弃：这时才有选错片的歧义。
                        assoc_rejected_ = true;
                        assoc_residual = best_cost;
                        assoc_residual_last_ = best_cost;
                        if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                            std::cerr << "[rune pick] association rejected: n=" << results.size()
                                      << " dt=" << dt << "s residual="
                                      << best_cost * 180.0 / CV_PI << "deg gate="
                                      << gate * 180.0 / CV_PI << "deg" << std::endl;
                        }
                        handleLost(timestamp);
                        return std::nullopt;
                    } else {
                        // 单片观测，但它离预测位置很远（≥ 一个槽位量级）。
                        const double observed = candidate_phase(results[index]);
                        const double step = 2.0 * CV_PI / 5.0;
                        if (std::abs(limitRad(observed - pending_switch_phase_)) < 0.35 * step) {
                            ++pending_switch_frames_;
                        } else {
                            pending_switch_phase_ = observed;
                            pending_switch_frames_ = 1;
                        }
                        const int confirm = std::max(1, assoc.switch_confirm_frames);
                        if (pending_switch_frames_ >= confirm) {
                            // 确认换片：把关联重置到新位置（角速度/方向保留），
                            assoc_phase_ = observed;
                            assoc_time_ = timestamp;
                            assoc_valid_ = true;
                            pending_switch_frames_ = 0;
                            assoc_choice = index;
                            assoc_used = true;
                            assoc_residual = best_cost;
                            if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                                std::cerr << "[rune pick] switch confirmed after "
                                          << confirm << " frames (residual "
                                          << best_cost * 180.0 / CV_PI << "deg)" << std::endl;
                            }
                        } else {
                            assoc_rejected_ = true;
                            assoc_residual = best_cost;
                            assoc_residual_last_ = best_cost;
                            if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                                std::cerr << "[rune pick] single-candidate jump pending ("
                                          << pending_switch_frames_ << "/" << confirm
                                          << "), residual " << best_cost * 180.0 / CV_PI
                                          << "deg gate " << gate * 180.0 / CV_PI << "deg"
                                          << std::endl;
                            }
                            handleLost(timestamp);
                            return std::nullopt;
                        }
                    }
                }
            }

            // 锁定：优先保持上一帧选中的那片扇叶，除非有候选"明显更优"并
            std::size_t chosen = best;
            if (assoc_used) {
                assoc_residual_last_ = assoc_residual;
                // 相位关联已经给出明确选择：不再让亮度/锁定滞回改判，
                chosen = assoc_choice;
                switch_votes_ = 0;
                picked_brightness_ = scores[chosen];
                if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                    std::cerr << "[rune pick] phase association: choice=" << chosen
                              << " residual=" << assoc_residual * 180.0 / CV_PI << " deg"
                              << " rate=" << assoc_rate_ << " rad/s"
                              << std::endl;
                }
            } else
            if (has_lock_) {
                int locked_index = -1;
                float locked_distance = std::numeric_limits<float>::max();
                for (std::size_t i = 0; i < results.size(); ++i) {
                    const cv::Point2f center = plateCenter(results[i]);
                    const float distance = cv::norm(center - locked_center_);
                    if (distance < locked_distance) {
                        locked_distance = distance;
                        locked_index = static_cast<int>(i);
                    }
                }
                const float lock_radius = static_cast<float>(config_.lock_break_px);
                const bool lock_found = locked_index >= 0 && locked_distance <= lock_radius;
                if (!lock_found) {
                    // 锁定目标这一帧没检出：短暂丢失就交给滤波滑行，不换靶；
                    if (++lock_miss_ >= config_.lock_max_miss) {
                        has_lock_ = false;
                        switch_votes_ = 0;
                        lock_miss_ = 0;
                    } else {
                        handleLost(timestamp);
                        return std::nullopt;
                    }
                } else {
                    // 已锁定：始终选离当前瞄准位置最近的那一片，不再比较亮度，
                    chosen = static_cast<std::size_t>(locked_index);
                    switch_votes_ = 0;
                }
                lock_miss_ = 0;
            }
            picked_brightness_ = scores[chosen];

            const cv::Point2f chosen_center = plateCenter(results[chosen]);
            locked_center_ = chosen_center;
            has_lock_ = true;
            target_class_ = results[chosen].class_id;

            // 更新相位关联状态：记下这一帧选中的相位与自估角速度。
            if (config_.phase_assoc.enabled && timestamp > 0.0) {
                const auto& assoc = config_.phase_assoc;
                assoc_residual_last_ = -1.0;
                const double phase = candidate_phase(results[chosen]);
                if (assoc_valid_) {
                    const double dt = timestamp - assoc_time_;
                    if (dt > 1e-4 && dt <= assoc.max_gap_s) {
                        const double observed_rate = limitRad(phase - assoc_phase_) / dt;
                        if (assoc_sign_ == 0 && std::abs(observed_rate) >= assoc.rate_min_rad_s) {
                            assoc_sign_ = observed_rate > 0.0 ? 1 : -1;
                        }
                        // 角速度只按幅值做 EMA 与夹取，方向单独用 assoc_sign_ 保持，
                        const double magnitude = std::clamp(
                            std::abs(observed_rate), assoc.rate_min_rad_s,
                            assoc.rate_max_rad_s);
                        const double smoothed = (1.0 - assoc.rate_smooth) *
                                                    std::abs(assoc_rate_) +
                                                assoc.rate_smooth * magnitude;
                        assoc_rate_ = (assoc_sign_ < 0 ? -1.0 : 1.0) * smoothed;
                    }
                } else {
                    // 首次捕获：方向未知（assoc_sign_=0），角速度先用标称值，
                    assoc_rate_ = assoc.nominal_rate_rad_s;
                    assoc_sign_ = 0;
                }
                assoc_phase_ = phase;
                assoc_time_ = timestamp;
                assoc_valid_ = true;
            }

            std::vector<RuneModel::Object> keep;
            keep.push_back(results[chosen]);
            results.swap(keep);
        }

        const auto attempt = [&](std::vector<RuneModel::Object>& candidates)
            -> std::optional<PowerRune> {
            std::vector<FanBlade> blades;
            blades.reserve(candidates.size());
            refine_stats_ = RefineStats{};
            refine_stats_.candidates = static_cast<int>(candidates.size());
            for (auto& candidate : candidates) {
                {
                    // 深大五点模型：图像上逆时针依次是 k0 -> k1 -> k4 -> k3，
                    if (candidate.keypoints.size() < 5) continue;
                    std::vector<cv::Point2f> points;
                    points.reserve(5);
                    for (const int index : config_.plate_point_indices) {
                        if (index < 0 ||
                            index >= static_cast<int>(candidate.keypoints.size())) {
                            points.clear();
                            break;
                        }
                        points.push_back(candidate.keypoints[static_cast<std::size_t>(index)]);
                    }
                    if (points.size() != 4) continue;
                    // 靶面几何精修（网络 ROI → 经典角点提取）：只在这个候选身上做，
                    if (config_.plate_refiner.enabled) {
                        ++refine_stats_.plate_refiner_tried;
                        const std::array<cv::Point2f, 4> network_points{points[0], points[1],
                                                                        points[2], points[3]};
                        const auto plate = plate_refiner_.refine(bgr_image, network_points);
                        refine_stats_.last_plate_area_ratio = plate.area_ratio;
                        refine_stats_.last_plate_size_ratio = plate.size_ratio;
                        if (plate.refined) {
                            points.assign(plate.points.begin(), plate.points.end());
                            ++refine_stats_.plate_refined;
                        }
                    }
                    points.push_back(candidate.center);
                    blades.emplace_back(points, candidate.center, Lit);
                    // 观测精修：轮廓几何 + 可用性闸门（config.refiner.enabled=false 时
                    cv::Point2f center = candidate.keypoints[2];
                    if (config_.refiner.enabled) {
                        RuneBladeRefinement detail;
                        center = refiner_.refineCenter(bgr_image, candidate.keypoints,
                                                       candidate.keypoints[2],
                                                       config_.our_color, &detail);
                        refine_stats_.last_armor_solidity = detail.armor_solidity;
                        refine_stats_.last_armor_area_ratio = detail.armor_area_ratio;
                        refine_stats_.last_armor_circularity = detail.armor_circularity;
                        refine_stats_.last_arm_solidity = detail.arm_solidity;
                        refine_stats_.last_arm_aspect = detail.arm_aspect;
                        refine_stats_.last_gap_px = detail.center_gap_px;
                        if (detail.usable) ++refine_stats_.usable;
                        if (detail.center_fused) ++refine_stats_.center_fused;
                        if (config_.refiner.require_usable && !detail.usable) {
                            // 几何不可信（灯臂轮廓破碎/长宽比离谱/装甲板凹陷）：
                            blades.pop_back();
                            ++refine_stats_.dropped;
                            continue;
                        }
                    }
                    blades.back().rune_center = center;
                    blades.back().has_rune_center = true;
                }
            }
            if (blades.empty()) return std::nullopt;
            last_blade_center_ = blades.front().center;
            last_blade_rmark_ = blades.front().has_rune_center ? blades.front().rune_center
                                                              : blades.front().center;

            const cv::Point2f center =
                estimateRuneCenter(blades, bgr_image, multi_blade_hub);
            // 自适应画布缩放的尺度：由靶面尺寸反推轨道半径。
            if (frame_orbit_px_ <= 5.0 && !blades.empty()) {
                const double orbit = cv::norm(blades.front().center - center);
                if (orbit > 5.0) frame_orbit_px_ = orbit;
            }
            PowerRune rune(blades, center, last_powerrune_);
            if (rune.is_unsolve()) return std::nullopt;
            return rune;
        };

        std::optional<PowerRune> rune = attempt(results);
        if (!rune.has_value() && config_.fallback_single_blade && results.size() > 1) {
            // Several blades are visible but they do not fit the lattice
            const auto best = std::max_element(
                results.begin(), results.end(),
                [](const RuneModel::Object& lhs, const RuneModel::Object& rhs) {
                    return lhs.prob < rhs.prob;
                });
            std::vector<RuneModel::Object> single;
            single.push_back(*best);
            rune = attempt(single);
        }

        if (!rune.has_value()) {
            handleLost(timestamp);
            return std::nullopt;
        }

        lifecycle_.noteAccepted(timestamp);
        last_eval_time_ = timestamp;
        last_powerrune_ = rune;
        // 自适应画布缩放用的尺度：本帧由靶面尺寸反推的轨道半径。用 EMA 平滑，
        if (frame_orbit_px_ > 5.0) {
            last_orbit_px_ = last_orbit_px_ > 5.0
                                 ? 0.7 * last_orbit_px_ + 0.3 * frame_orbit_px_
                                 : frame_orbit_px_;
        }
        if (target_class_ < 0) {
            // prefer_bright_blade 关闭时上面没有选靶分支，这里按"离实际瞄准的
            float best_distance = std::numeric_limits<float>::max();
            for (const RuneModel::Object& candidate : results) {
                const float distance = cv::norm(candidate.center - rune->target().center);
                if (distance < best_distance) {
                    best_distance = distance;
                    target_class_ = candidate.class_id;
                }
            }
        }
        return rune;
    }

    void RuneDetector::updateCanvasScale(const cv::Mat& bgr_image)
    {
        if (!config_.canvas_adaptive || bgr_image.empty()) return;
        const int input_width = config_.model.input_size;
        const int input_height =
            config_.model.input_height > 0 ? config_.model.input_height : input_width;
        if (input_width <= 0 || input_height <= 0) return;

        // 画布缩放系数 cs 的含义：原图先被缩放/填充成 cs 倍大小，再 letterbox 到
        const double base = std::min(static_cast<double>(input_width) / bgr_image.cols,
                                     static_cast<double>(input_height) / bgr_image.rows);
        double scale = 0.0;
        if (last_orbit_px_ > 5.0 && lost_frames_ < std::max(1, config_.canvas_sweep_after_lost)) {
            scale = last_orbit_px_ * base / std::max(8.0, config_.canvas_target_orbit_px);
        } else if (!config_.canvas_sweep_scales.empty()) {
            // 还没有尺度、或者连续丢帧太久：按阶梯轮流试，重新捕获后就交回上面的规则。
            const std::size_t count = config_.canvas_sweep_scales.size();
            scale = config_.canvas_sweep_scales[static_cast<std::size_t>(canvas_sweep_index_) % count];
            ++canvas_sweep_index_;
        } else {
            scale = config_.model.input_pad_scale;
        }
        canvas_scale_ = std::clamp(scale, config_.canvas_scale_min, config_.canvas_scale_max);
        model_.setCanvasScale(canvas_scale_);
    }

    void RuneDetector::prepareModelInput(const cv::Mat& bgr_image)
    {
        updateCanvasScale(bgr_image);
        const bool have_recenter =
            config_.canvas_recenter && last_powerrune_.has_value() &&
            canvas_scale_ <= config_.canvas_recenter_max_scale &&
            lost_frames_ < std::max(1, config_.canvas_sweep_after_lost);
        if (have_recenter) {
            model_.setCanvasCenter(last_powerrune_->r_center, true);
        } else {
            model_.setCanvasCenter({0.0f, 0.0f}, false);
        }
    }

    std::vector<RuneModel::Object> RuneDetector::detectModelObjects(
        const cv::Mat& bgr_image, bool best_only)
    {
        prepareModelInput(bgr_image);
        std::vector<RuneModel::Object> objects =
            best_only ? model_.detectBest(bgr_image) : model_.detect(bgr_image);
        last_roi_active_ = model_.lastRoiActive();
        last_roi_crop_ = model_.lastRoiCrop();
        last_roi_fallback_ = false;

        // ROI 裁剪把视野缩小了。如果这一帧一个候选都没有，立即用整幅图重跑
        if (objects.empty() && last_roi_active_ && config_.canvas_recenter_fallback) {
            last_roi_fallback_ = true;
            model_.setCanvasCenter({0.0f, 0.0f}, false);
            objects = best_only ? model_.detectBest(bgr_image) : model_.detect(bgr_image);
            last_roi_active_ = model_.lastRoiActive();
            last_roi_crop_ = model_.lastRoiCrop();
        }
        return objects;
    }

    std::optional<PowerRune> RuneDetector::detect(const cv::Mat& bgr_image, double timestamp)
    {
        const double effective_timestamp =
            timestamp > 0.0 ? timestamp : StandardClock::nowSeconds();
        std::vector<RuneModel::Object> objects = detectModelObjects(bgr_image, false);
        std::optional<PowerRune> rune =
            buildRune(std::move(objects), bgr_image, effective_timestamp);
        lost_frames_ = rune.has_value() ? 0 : lost_frames_ + 1;
        return rune;
    }

    std::optional<PowerRune> RuneDetector::detectBest(const cv::Mat& bgr_image, double timestamp)
    {
        const double effective_timestamp =
            timestamp > 0.0 ? timestamp : StandardClock::nowSeconds();
        std::vector<RuneModel::Object> objects = detectModelObjects(bgr_image, true);
        std::optional<PowerRune> rune =
            buildRune(std::move(objects), bgr_image, effective_timestamp);
        lost_frames_ = rune.has_value() ? 0 : lost_frames_ + 1;
        return rune;
    }
} // namespace auto_aim::energy
