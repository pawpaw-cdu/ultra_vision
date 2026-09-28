#include "auto_buff/rune_detector.hpp"
#include "auto_buff/support/math.hpp"

#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <utility>

namespace auto_aim::energy
{
    RuneDetector::RuneDetector(const RuneDetectorConfig& config)
        : config_(config), model_(config.model), refiner_(config.refiner),
          plate_refiner_(config.plate_refiner), blade_state_(config.blade_state)
    {
        // 启动横幅：把"这次到底用了哪套模型/阈值"打出来 —— 现场排查最常见的
        // 误判就是配置与模型不匹配（类别数、点序、输入尺寸、阈值），
        // 所以这几个量必须在日志开头就有据可查。
        std::cout << "[rune config] output_layout=" << config.model.output_layout
                  << " classes=" << config.model.num_classes
                  << " keypoints=" << config.model.num_keypoints
                  << (config.model.keypoints_have_confidence ? "(with conf)" : "")
                  << " input=" << config.model.input_size << "x"
                  << (config.model.input_height > 0 ? config.model.input_height
                                                    : config.model.input_size)
                  << " pad_scale=" << config.model.input_pad_scale
                  << " input_scale=" << config.model.input_scale
                  << " score_thr=" << config.model.confidence_threshold
                  << " kpt_conf=" << config.model.keypoint_confidence_threshold
                  << " min_kpt=" << config.model.min_valid_keypoints
                  << " require_inactive_class=" << (config.require_inactive_class ? 1 : 0)
                  << " pad_adaptive=" << (config.canvas_adaptive ? 1 : 0)
                  << " pad_target_orbit_px=" << config.canvas_target_orbit_px
                  << " model=" << config.model.model_path << std::endl;
    }

    void RuneDetector::handleLost()
    {
        ++lost_;
        if (lost_ >= config_.max_lost_frames) {
            status_ = RuneTrackStatus::Lost;
            last_powerrune_ = std::nullopt;
            return;
        }
        status_ = RuneTrackStatus::TemporaryLost;
    }

    cv::Point2f RuneDetector::estimateRuneCenter(std::vector<FanBlade>& blades,
                                                const cv::Mat& bgr_image,
                                                const std::optional<cv::Point2f>& multi_blade_hub)
    {
        if (blades.empty()) return {0.0f, 0.0f};

        // 0) 多片扇叶求交得到的圆心最可信（每片都给出一条"圆心→靶心"射线，
        //    射线交点与各片的 R 标点无关，噪声被平均掉）。
        if (multi_blade_hub.has_value()) {
            hub_source_ = HubSource::MultiBladeRay;
            return multi_blade_hub.value();
        }

        // 靶面点半径（像素）：四个靶面点相对靶心的平均距离。圆心一定在
        // "靶心 + N 倍该半径" 的环上（N = 圆心到靶心 / 靶面点半径）。
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
        //    比"靶心 + 臂点外推"稳得多。仿真渲染和训练域差得远时，网络会把
        //    它压到靶心附近，这种"圆心 = 靶心"的点必须丢掉。
        cv::Point2f hint(0.0f, 0.0f);
        int hint_count = 0;
        const double orbit_ratio = std::max(1.5, config_.center_orbit_ratio);
        for (const FanBlade& blade : blades) {
            if (!blade.has_rune_center) continue;
            const double offset = cv::norm(blade.rune_center - blade.center);
            // 圆心到靶心的像素距离应当 ≈ orbit_ratio × 靶面点半径：太近说明网络把
            // 第 5 点压在了靶心上（仿真域差异时常见），太远说明那是个野点——离域
            // 画面上网络会在画面边缘给出高置信度的假点，直接用它会把 PnP 选解、
            // 相位和瞄准全部带偏（实测 EKF 的 yaw 会跑到 30° 以上）。
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
        // 用它（RP-26Rune：rune_center = point_R）。实测在归一化输入下它与
        // 画面里 5 条臂的交汇点误差 0.5 px 量级，深度校正全靠它，直接用。
        if (have_hint) {
            hub_source_ = HubSource::RMarkKeypoint;
            return hint;
        }

        // 2) 旧六点布局：第 6 个关键点是臂上的点，沿"臂点 - 靶心"外推。
        cv::Point2f rough(0.0f, 0.0f);
        bool have_rough = false;
        if (config_.model.output_layout != "v8_pose_5kpt") {
            for (const FanBlade& blade : blades) {
                if (blade.points.size() < 6) continue;
                const cv::Point2f plate_center = blade.points[4];
                const cv::Point2f arm_point = blade.points[5];
                rough += (arm_point - plate_center) *
                             static_cast<float>(config_.center_extrapolation) +
                         plate_center;
                have_rough = true;
            }
            if (have_rough) rough /= static_cast<float>(blades.size());
        }

        cv::Mat gray;
        cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);

        // 3) 没有可用方向：沿环带找"最亮方位"。能量机关的 R 标是画面里唯一
        //    稳定的亮点（仿真里是一块红色 R），靶面图案离该环带也够远。
        if (!have_rough) {
            const cv::Point2f plate_center = blades.front().center;
            // 判据是"小亮点 + 周围暗"：R 标是几像素大小的亮块（仿真是纯红，
            // 灰度只有 ~76，二值化阈值抓不到，必须用通道最大值），而靶面图案
            // 是个大亮块，两者用"小窗峰值 − 大窗均值"区分得开。
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

        // The model's keypoints 5 and 6 are the plate centre and a point on
        // the arm; the rune centre lies on that line, a fixed multiple of the
        // plate-centre-to-arm distance beyond the plate centre.
        // Refine on the rune's bright hub: threshold, dilate, then keep the
        // blob nearest to the rough estimate whose aspect ratio is closest to 1.
        cv::Mat binary;
        cv::threshold(gray, binary, config_.gray_threshold, 255, cv::THRESH_BINARY);
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(config_.dilate_size, config_.dilate_size));
        cv::Mat dilated;
        cv::dilate(binary, dilated, kernel, cv::Point(-1, -1), 1);

        const double radius =
            cv::norm(blades.front().points[2] - blades.front().center) * config_.center_mask_ratio;
        cv::Mat mask = cv::Mat::zeros(dilated.size(), CV_8U);
        cv::circle(mask, rough, radius, cv::Scalar(255), -1);
        cv::bitwise_and(dilated, mask, dilated);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::Point2f center = rough;
        double best_score = std::numeric_limits<double>::max();
        for (const auto& contour : contours) {
            const cv::RotatedRect rect = cv::minAreaRect(contour);
            if (rect.size.width <= 0.0f || rect.size.height <= 0.0f) continue;
            double score = std::max(rect.size.width, rect.size.height) /
                           std::min(rect.size.width, rect.size.height);
            score += cv::norm(rect.center - rough) / std::max(1.0, radius / 3.0);
            if (score < best_score) {
                best_score = score;
                center = rect.center;
            }
        }
        hub_source_ = HubSource::RingBrightness;
        return center;
    }

    std::optional<PowerRune> RuneDetector::buildRune(std::vector<RuneModel::Object>&& results,
                                                     const cv::Mat& bgr_image, double timestamp)
    {
        // 类别统计（CSV 与激活状态机用）：0=未激活，1=小符已激活，2=大符已激活。
        class_counts_ = {0, 0, 0};
        // 诊断量每帧清零：否则没建出 rune 的帧会沿用上一帧的值，
        // 和同一行的观测列（此时是 0）混在一起，会得出"hub 偏了 300px"这种假结论。
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
        const bool pose_layout = config_.model.output_layout == "v8_pose_5kpt";
        const auto plateCenter = [](const RuneModel::Object& candidate) {
            return candidate.center;
        };

        // ① 几何修正：每片扇叶（含已激活）都给出"圆心→靶心"方向，圆心必在这些
        //    射线上。≥2 片时用最小二乘求交点 —— 已激活扇叶在这里只贡献几何约束，
        //    不参与目标选择。这正是"用已激活扇叶做重投影修正"的最小实现。
        std::optional<cv::Point2f> multi_blade_hub;
        if (pose_layout) {
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
        //    已激活扇叶之间来回换靶（现场实测的"云台左右拉扯"）。
        //    但离线验证（录制帧逐帧）发现：网络会把**点亮图案**误判成 class 1
        //    （88 个有候选帧里 20 帧，23%）。硬过滤会在这些帧丢掉真正的目标、
        //    转去锁更暗的片子（配对 A/B：每轮命中 30% vs 放开类别 36%）。
        //    折中：默认只在 class 0 里选；只有当某个非 class0 候选的亮点分
        //    **明显更亮**（超过最佳 class0 候选 margin 倍）时才采信它 ——
        //    这既救回"点亮片被误分类"的帧，又不会去锁已经打过的亮片。
        // 候选的"图像相位"（圆心→靶面中心的方向）。类别闸门与相位关联都用它，
        // 所以在函数作用域里定义一次。
        const auto candidate_phase = [&](const RuneModel::Object& candidate) -> double {
            if (candidate.keypoints.size() < 5) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const cv::Point2f hub = candidate.keypoints[2];
            const cv::Point2f plate = plateCenter(candidate);
            return std::atan2(plate.y - hub.y, plate.x - hub.x);
        };
        // ---- 几何连续优先（必须在类别过滤**之前**）-----------------------------
        // 2026-09-24 实测的"最后一片打不出去"根因：只剩一片未激活时，网络会把
        // **已激活片**标成 class 0、把真正的目标标成 class 1；类别过滤按类别重排
        // 候选，于是"几何上我们一直在跟踪的那片"被挤掉，瞄点被拖到已激活片上
        // （仿真事件里直接可见：highlighted=1 时我们连续 4 发打在 hit=3，
        // scored 全 0，本轮随后超时失败）。
        //
        // 三家参考都不允许"类别"决定目标：sp_vision 用"亮片数 + 位置连续性"
        // （buff_type.cpp 的 PowerRune），rm_vision_core 在 inactive 为空时本帧
        // 不更新、并把 active 集合只当几何参考，RP-26Rune 用整符模型做配准。
        // 这里做等价的事：先用**运动模型预测的相位**在全部候选里找"连续的那片"，
        // 找到就只保留它，类别随后不再参与。
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
                          << results.front().class_id << "), class filter skipped"
                          << std::endl;
            }
        }
        if (pose_layout && config_.require_inactive_class) {
            std::vector<RuneModel::Object> inactive;
            inactive.reserve(results.size());
            for (const RuneModel::Object& candidate : results) {
                if (candidate.class_id == 0) inactive.push_back(candidate);
            }
            // 锁定的那片一旦变成"已激活"，说明刚打中它：立刻解锁定。
            if (has_lock_) {
                for (const RuneModel::Object& candidate : results) {
                    if (candidate.class_id == 0) continue;
                    if (cv::norm(candidate.center - locked_center_) <= config_.lock_break_px) {
                        has_lock_ = false;
                        lock_miss_ = 0;
                        break;
                    }
                }
            }
            // 亮点判据（与 prefer_bright_blade 同一套：小窗峰值 − 大窗均值）
            // bright 图**一次**算好：原来这条 lambda 每调用一次都重做一遍整幅
            // cv::split + 两次 cv::max（640×480×3 ≈ 0.9 MB/次，一帧 5~10 个候选），
            // 是帧循环里"非推理开销 ~25 ms"的嫌疑之一（见 docs §11）。
            cv::Mat channels[3];
            cv::split(bgr_image, channels);
            const cv::Mat bright_image = cv::max(cv::max(channels[0], channels[1]), channels[2]);
            const auto contrast = [&](const RuneModel::Object& candidate) {
                const cv::Point2f center = candidate.center;
                const float half = 0.5f * static_cast<float>(config_.blade_roi_ratio) *
                                   std::max(candidate.rect.width, candidate.rect.height);
                const int x = static_cast<int>(center.x);
                const int y = static_cast<int>(center.y);
                cv::Rect small_roi(x - 2, y - 2, 5, 5);
                cv::Rect large_roi(x - static_cast<int>(half), y - static_cast<int>(half),
                                   std::max(2, static_cast<int>(2 * half)),
                                   std::max(2, static_cast<int>(2 * half)));
                small_roi &= cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
                large_roi &= cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
                if (small_roi.width <= 0 || large_roi.width <= 0) return 0.0;
                double minimum = 0.0;
                double peak = 0.0;
                cv::minMaxLoc(bright_image(small_roi), &minimum, &peak);
                return peak - cv::mean(bright_image(large_roi))[0];
            };
            double best_inactive = -1.0;
            for (const RuneModel::Object& candidate : inactive) {
                best_inactive = std::max(best_inactive, contrast(candidate));
            }
            double best_other = -1.0;
            const RuneModel::Object* brightest_other = nullptr;
            for (const RuneModel::Object& candidate : results) {
                if (candidate.class_id == 0) continue;
                const double score = contrast(candidate);
                if (score > best_other) {
                    best_other = score;
                    brightest_other = &candidate;
                }
            }
            if (config_.rescue_missing_inactive && brightest_other != nullptr &&
                best_other > 0.0 &&
                best_other > config_.inactive_class_margin * std::max(0.0, best_inactive)) {
                // 类别证据**不能单独**把跟踪从几何上连续的那片拉走：已激活片变多
                // 以后（尤其待激活片夹在两片已激活片之间时）网络的 class 0/1 会在
                // 相邻片之间翻烧饼，只看亮度+类别就会逐帧换靶 —— 现场表现就是
                // 云台左右甩、槽位记账被污染。
                // 闸门：只有当这个非 class0 候选**比 class0 候选更靠近运动模型的
                // 预测位置**时，才认"点亮片被误分类"。
                bool continuity_ok = true;
                if (assoc_valid_ && timestamp > 0.0) {
                    const double dt = timestamp - assoc_time_;
                    if (dt > 1e-4 && dt <= config_.phase_assoc.max_gap_s) {
                        const double predicted = assoc_phase_ + assoc_rate_ * dt;
                        const double other_phase = candidate_phase(*brightest_other);
                        double other_residual = std::numeric_limits<double>::max();
                        if (std::isfinite(other_phase)) {
                            other_residual = std::abs(limitRad(other_phase - predicted));
                        }
                        double inactive_residual = std::numeric_limits<double>::max();
                        for (const RuneModel::Object& candidate : inactive) {
                            const double phase = candidate_phase(candidate);
                            if (!std::isfinite(phase)) continue;
                            inactive_residual =
                                std::min(inactive_residual,
                                         std::abs(limitRad(phase - predicted)));
                        }
                        // 非 class0 候选必须明显更贴近预测（>0.15 rad ≈ 8.6°）才算数。
                        continuity_ok = other_residual < inactive_residual - 0.15;
                    }
                }
                if (continuity_ok) {
                    // 点亮片被误分类：只保留这**一个**候选，避免同时把别的亮片也放进来
                    std::vector<RuneModel::Object> picked;
                    picked.push_back(*brightest_other);
                    results.swap(picked);
                    if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                        std::cerr << "[rune pick] class!=0 candidate accepted by brightness margin ("
                                  << best_other << " vs " << best_inactive
                                  << ", geometry-continuous)" << std::endl;
                    }
                } else {
                    results.swap(inactive);
                    if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                        std::cerr << "[rune pick] class!=0 candidate rejected by continuity gate ("
                                  << best_other << " vs " << best_inactive << ")" << std::endl;
                    }
                }
            } else {
                results.swap(inactive);
            }
            if (results.empty()) {
                handleLost();
                return std::nullopt;
            }
        }

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
        // are visible at once (the unlit artwork is drawn dark), and picking by
        // network score alone made the tracker lock onto a dark blade, which is
        // exactly the "hits the neighbouring target" failure.
        // 注意：这里必须对"只检出一片"的帧也生效。大符每轮点亮两片，关键点
        // 网络经常一帧只出其中一片、下一帧换成另一片；如果只在多候选时做锁定，
        // 单候选帧就会各自为政，瞄点仍在两片之间来回抖。
        if (config_.prefer_bright_blade && !results.empty()) {
            // 亮点判据用"小窗峰值 − 大窗均值"，而不是区域均值：仿真里画面
            // 底部的白色反光整片都很亮，均值排序会把它排在真正点亮的靶面前面
            // （网络也确实会在那片反光上出框，距离解算因此跑到十几米）。
            // 点亮的靶面是"小亮块 + 周围暗"，反光是"大面积均匀亮"，两者可分。
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
                // 会把暗扇叶的轮廓也平均进来）；大窗取候选框本身，用"小窗峰值 −
                // 大窗均值"衡量"小亮块压在场地上"的程度。这套参数在仿真实测里
                // 选靶准确率 ~78%（对比按网络置信度只有 29%）。
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
            // 只排除明显是暗色图案的误检）。
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
            //
            // 大符每轮两片同时点亮，两片相位差 72°/144°。按亮度竞争时两片亮度
            // 接近就会逐帧易主：观测相位出现 ±130° 的跳变，EKF 的角速度通道
            // 被打到 ±95 rad/s（物理上只可能 0~2.09），瞄准角随之前后甩 ——
            // 现场表现就是"在两块待激活扇叶之间高频抖动"。
            //
            // 做法借深大 SmallRuneKalmanFilter：用"上一次关联到的相位 +
            // 自估角速度 × dt"预测本帧相位，取最接近的候选；残差超过容差就
            // 认为本帧没有可信观测（返回空，让滤波滑行），把跳片挡在 EKF 之前。
            // 自估角速度用相邻两次接受观测的相位差做 EMA，并夹在大符物理区间里。
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
                    //   predicted = 上次相位 + 自估角速度 × dt
                    //   选 |wrap(phase_i - predicted)| 最小的候选；残差超过门限就
                    //   认为本帧没有可信观测。
                    // 为什么不直接用"隐含角速度必须落在物理区间"：小 dt 时相位
                    // 测量噪声（±0.1~0.6 rad）折算出的角速度噪声可达 ±20 rad/s，
                    // 物理区间会把 40% 的合法观测误杀（实测）。残差门限与 dt 无关，
                    // 而两片扇叶相隔 1.26 rad，门限 0.6 rad 足以区分。
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
                        // 单片（网络本帧只给一片）没有歧义，交给锁定/状态估计的
                        // 换叶确认去处理，否则会把 70% 的单片观测白白丢成滑行
                        // （实测单候选帧占拒绝的 90%）。
                        // 所有候选都离预测相位太远：本帧观测不可信（跳片/误检），
                        // 交给状态估计滑行。
                        assoc_rejected_ = true;
                        assoc_residual = best_cost;
                        assoc_residual_last_ = best_cost;
                        if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                            std::cerr << "[rune pick] association rejected: n=" << results.size()
                                      << " dt=" << dt << "s residual="
                                      << best_cost * 180.0 / CV_PI << "deg gate="
                                      << gate * 180.0 / CV_PI << "deg" << std::endl;
                        }
                        handleLost();
                        return std::nullopt;
                    } else {
                        // 单片观测，但它离预测位置很远（≥ 一个槽位量级）。
                        //
                        // 2026-09-24 实测：已激活片变多以后，网络**逐帧**在相邻亮片
                        // 之间翻烧饼 —— 尤其"待激活片夹在两片已激活片之间"时，
                        // 观测相位在 ±72°/144° 之间来回跳（30 s 里 26 次 >40° 跳变，
                        // 其中一段 1 s 内跳 5 次），而且这些帧大多是**单候选**，
                        // 所以旧的"单片无歧义、直接放行"规则会把跳变原样喂给估计器：
                        // EKF 的 roll/角速度被反复拽，瞄点在两片之间甩、记账也被污染。
                        //
                        // 判据只有一个：**连续确认**。连续 switch_confirm_frames 帧
                        // 看到同一个新位置（互相在 0.35 槽位内）才认这次换片；确认之前
                        // 本帧不产生观测，交给估计器按运动模型滑行（滑行的小符相位是
                        // 可靠的：转速固定 60°/s）。真实换片是持续事件，两帧就够区分，
                        // 代价只有 ~66 ms 的滑行。
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
                            // 本帧照常产出观测。
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
                            handleLost();
                            return std::nullopt;
                        }
                    }
                }
            }

            // 锁定：优先保持上一帧选中的那片扇叶，除非有候选"明显更优"并
            // 连续多帧确认。没有这层滞回时，亮度接近的两个候选会逐帧易主，
            // 目标角在 72° 两档之间来回跳，云台看起来就是左右拉扯。
            std::size_t chosen = best;
            if (assoc_used) {
                assoc_residual_last_ = assoc_residual;
                // 相位关联已经给出明确选择：不再让亮度/锁定滞回改判，
                // 避免两片之间来回切。锁定解禁仍由上面的"锁定片变 class!=0"负责。
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
                    // 连续丢失超过阈值才允许重新选，避免被别片抢走。
                    if (++lock_miss_ >= config_.lock_max_miss) {
                        has_lock_ = false;
                        switch_votes_ = 0;
                        lock_miss_ = 0;
                    } else {
                        handleLost();
                        return std::nullopt;
                    }
                } else {
                    // 已锁定：始终选离当前瞄准位置最近的那一片，不再比较亮度，
                    // 否则大符两片同时点亮时会来回跳。
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
                        // 这样顺时针（负角速度）不会被 clamp 到正区间里去。
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
                    // 下一帧由隐含角速度确定方向。
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
                if (pose_layout) {
                    // 深大五点模型：图像上逆时针依次是 k0 -> k1 -> k4 -> k3，
                    // 对应 PnP 物点的 外 -> 右 -> 内 -> 左（RuneSolver 的顺序），
                    // k2 是 R 标（机关旋转中心）。
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
                    // 不通过一致性检查就原样保留网络结果（见 rune_plate_refiner.hpp）。
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
                    // SCUT 式经典状态特征：在网络四点外接框（放大 25%）内统计
                    // "点亮像素"（红/橙色 LED 图案：R 明显大于 B）的面积与占比。
                    {
                        const cv::Rect box = cv::boundingRect(points);
                        const int margin_x = static_cast<int>(0.25 * box.width);
                        const int margin_y = static_cast<int>(0.25 * box.height);
                        cv::Rect roi(box.x - margin_x, box.y - margin_y,
                                     box.width + 2 * margin_x, box.height + 2 * margin_y);
                        roi &= cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
                        if (roi.width > 4 && roi.height > 4) {
                            std::vector<cv::Mat> channels;
                            cv::split(bgr_image(roi), channels);
                            cv::Mat lit;
                            cv::subtract(channels[2], channels[0], lit);      // R − B（红/橙图案）
                            cv::threshold(lit, lit, 40, 255, cv::THRESH_BINARY);
                            const double area = cv::countNonZero(lit);
                            refine_stats_.classic_lit_area = area;
                            refine_stats_.classic_lit_ratio =
                                area / static_cast<double>(roi.width * roi.height);
                        }
                    }
                    // 经典状态分类（网络管召回、经典管状态）：圆心用当前估计的 R 标，
                    // 轨道半径由**本片自己的靶面尺寸**反推（两者同一帧、尺度自洽）。
                    // 注意不要用 hub_source_ / last_blade_* 这些"上一帧的"状态：
                    // 那个量在 hub 不可用时退化成 8 px，ROI 变成靶心上的 12×12 小块，
                    // lit_ratio 随之变成一个几乎恒定的假数（§46/§47 实测 0.23~0.29
                    // 与真实状态无关）。
                    {
                        double plate_radius = 0.0;
                        for (const int index : config_.plate_point_indices) {
                            if (index < 0 || index >= static_cast<int>(candidate.keypoints.size())) {
                                continue;
                            }
                            plate_radius +=
                                cv::norm(candidate.keypoints[static_cast<std::size_t>(index)] -
                                         candidate.center);
                        }
                        plate_radius /= 4.0;
                        const double orbit =
                            std::max(8.0,
                                     plate_radius /
                                         std::max(0.05, config_.blade_state.plate_orbit_ratio));
                        // 灯臂采样要一个"朝圆心"的方向：优先用本片自己的 k2（同一帧、
                        // 同一片），它在几何上不合理（塌到靶心 / 太远）时才退回上一帧
                        // 的估计，最后才退回靶心本身（那时 arm 量会失效，只当参考）。
                        cv::Point2f hub_hint = candidate.center;
                        if (candidate.keypoints.size() >= 5) {
                            const cv::Point2f k2 = candidate.keypoints[2];
                            const double offset = cv::norm(k2 - candidate.center);
                            if (offset > 0.5 * plate_radius && offset < 1.5 * orbit) {
                                hub_hint = k2;
                            } else if (cv::norm(last_blade_rmark_ - candidate.center) >
                                       1.5 * plate_radius) {
                                hub_hint = last_blade_rmark_;
                            }
                        }
                        const auto state = blade_state_.classify(bgr_image, hub_hint,
                                                                 candidate.center, orbit);
                        frame_orbit_px_ = frame_orbit_px_ > 0.0
                                              ? 0.5 * (frame_orbit_px_ + orbit)
                                              : orbit;
                        refine_stats_.state_lit_ratio = state.lit_ratio;
                        refine_stats_.state_arm_ratio = state.arm_lit_ratio;
                        refine_stats_.state_code = static_cast<int>(state.state);
                        refine_stats_.state_orbit_px = state.orbit_px;
                        refine_stats_.state_lit_area = state.lit_area;
                        refine_stats_.state_roi_px = state.roi_px;
                        if (std::getenv("ULTRA_VISION_RUNE_DEBUG")) {
                            std::cerr << "[rune state] " << RuneBladeState::name(state.state)
                                      << " orbit=" << state.orbit_px << " roi=" << state.roi_px
                                      << " lit_area=" << state.lit_area
                                      << " lit_ratio=" << state.lit_ratio
                                      << " arm=" << state.arm_lit_ratio << " hub_src="
                                      << static_cast<int>(hub_source_) << " hub=(" << hub_hint.x
                                      << "," << hub_hint.y << ") blade=(" << candidate.center.x
                                      << "," << candidate.center.y << ")" << std::endl;
                        }
                    }
                    points.push_back(candidate.center);
                    blades.emplace_back(points, candidate.center, Lit);
                    // 观测精修：轮廓几何 + 可用性闸门（config.refiner.enabled=false 时
                    // 下面全部退化成"原样使用 k2"，故默认路径与之前完全一致）。
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
                            // 这一片本帧不参与解算，宁可让滤波器滑行。
                            blades.pop_back();
                            ++refine_stats_.dropped;
                            continue;
                        }
                    }
                    blades.back().rune_center = center;
                    blades.back().has_rune_center = true;
                } else {
                    if (candidate.keypoints.size() < 6) continue;
                    blades.emplace_back(candidate.keypoints, candidate.keypoints[4], Lit);
                }
            }
            if (blades.empty()) return std::nullopt;
            last_blade_center_ = blades.front().center;
            last_blade_rmark_ = blades.front().has_rune_center ? blades.front().rune_center
                                                              : blades.front().center;

            const cv::Point2f center =
                estimateRuneCenter(blades, bgr_image, multi_blade_hub);
            // 自适应画布缩放的尺度：五点布局在前面已由靶面尺寸反推；旧六点布局
            // （没有靶面四点）退化成"圆心→靶心"的像素距离，同样够用。
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
            // (recorded footage where the whole rune glows). Track the most
            // confident one, which is what the reference implementation does.
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
            handleLost();
            return std::nullopt;
        }

        status_ = RuneTrackStatus::Track;
        lost_ = 0;
        last_powerrune_ = rune;
        // 自适应画布缩放用的尺度：本帧由靶面尺寸反推的轨道半径。用 EMA 平滑，
        // 单帧的靶面抖动（角点噪声）不该让缩放系数来回跳。
        if (frame_orbit_px_ > 5.0) {
            last_orbit_px_ = last_orbit_px_ > 5.0
                                 ? 0.7 * last_orbit_px_ + 0.3 * frame_orbit_px_
                                 : frame_orbit_px_;
        }
        if (target_class_ < 0) {
            // prefer_bright_blade 关闭时上面没有选靶分支，这里按"离实际瞄准的
            // 那片最近"补一次类别，好让 CSV 与激活状态机拿到一致的类别。
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
        // 网络输入。因此"机关在网络输入里的轨道半径" = orbit_px × min(iw/W, ih/H) / cs，
        // 反解 cs 就让机关落在训练尺度上（canvas_target_orbit_px）。
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

    std::optional<PowerRune> RuneDetector::detect(const cv::Mat& bgr_image, double timestamp)
    {
        updateCanvasScale(bgr_image);
        std::optional<PowerRune> rune = buildRune(model_.detect(bgr_image), bgr_image, timestamp);
        lost_frames_ = rune.has_value() ? 0 : lost_frames_ + 1;
        return rune;
    }

    std::optional<PowerRune> RuneDetector::detectBest(const cv::Mat& bgr_image, double timestamp)
    {
        updateCanvasScale(bgr_image);
        std::optional<PowerRune> rune = buildRune(model_.detectBest(bgr_image), bgr_image, timestamp);
        lost_frames_ = rune.has_value() ? 0 : lost_frames_ + 1;
        return rune;
    }
} // namespace auto_aim::energy
