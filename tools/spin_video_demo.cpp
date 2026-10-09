// 实拍视频上的小陀螺灯条轨迹几何 demo：
//   检测(NN) → 跟踪一块板的两根灯条 → 多灯条几何拟合 → 输出叠加视频。
// 相机内参未知：默认按常见手机主摄 FOV≈66° 估计 fx=fy=0.77*width，可用参数覆盖。
//
// 用法:
//   spin_video_demo <video> <model.xml> <out.mp4> [fx] [cx] [cy] [c] [L] [max_frames]
//   fx<=0 用默认(0.77*width)；cx/cy<=0 用图像中心。
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "perception/nn_detector.hpp"
#include "estimation/spin_geometry_estimator.hpp"

namespace
{
    auto_aim::SpinBarObservation makeObs(double t, int plate, int side,
                                         const cv::Point2f& top, const cv::Point2f& bot)
    {
        auto_aim::SpinBarObservation o;
        o.t = t; o.plate = plate; o.side = side;
        o.u_top = top.x; o.v_top = top.y;
        o.u_bot = bot.x; o.v_bot = bot.y;
        return o;
    }
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: spin_video_demo <video> <model.xml> <out.mp4> "
                     "[fx] [cx] [cy] [c] [L] [max_frames]\n";
        return 2;
    }
    const std::string video = argv[1];
    const std::string model = argv[2];
    const std::string out_path = argv[3];

    cv::VideoCapture cap(video);
    if (!cap.isOpened()) { std::cerr << "cannot open video\n"; return 1; }
    const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS) > 1.0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;

    double fx = (argc > 4) ? std::atof(argv[4]) : 0.0;
    double cx = (argc > 5) ? std::atof(argv[5]) : 0.0;
    double cy = (argc > 6) ? std::atof(argv[6]) : 0.0;
    const double c_off = (argc > 7) ? std::atof(argv[7]) : 0.0655;
    const double L_len = (argc > 8) ? std::atof(argv[8]) : 0.055;
    const int max_frames = (argc > 9) ? std::atoi(argv[9]) : 100000;
    if (fx <= 0.0) fx = 0.77 * W;
    if (cx <= 0.0) cx = 0.5 * W;
    if (cy <= 0.0) cy = 0.5 * H;

    std::cout << "video " << W << "x" << H << "@" << fps << "  K: fx=fy=" << fx
              << " cx=" << cx << " cy=" << cy << "  c=" << c_off << " L=" << L_len << "\n";

    auto_aim::NnDetectorConfig det_cfg;
    det_cfg.model_path = model;
    det_cfg.device = "CPU";
    det_cfg.output_format = "rp24";
    det_cfg.score_threshold = 0.25;
    det_cfg.min_confidence = 0.25;
    det_cfg.filter_by_color = false;
    auto_aim::NnArmorDetector detector(det_cfg);

    auto_aim::SpinGeometryConfig gcfg;
    gcfg.K.fx = fx; gcfg.K.fy = fx; gcfg.K.cx = cx; gcfg.K.cy = cy;
    gcfg.bar_offset_c = c_off;
    gcfg.bar_length_L = L_len;
    gcfg.distance_hint = 2.5;
    gcfg.estimate_axis = true;

    cv::VideoWriter writer(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(W, H));
    if (!writer.isOpened())
    {
        // 退回 avc1
        writer.open(out_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(W, H));
    }

    std::vector<auto_aim::SpinBarObservation> obs;
    auto_aim::SpinGeometryResult last;
    // 导出被跟踪灯条的端点，供局部拟合器离线使用。
    std::ofstream bars_csv(out_path + ".bars.csv");
    bars_csv << "t,u_lt,v_lt,u_lb,v_lb,u_rt,v_rt,u_rb,v_rb\n";
    bars_csv << std::fixed << std::setprecision(3);
    // 稳像：帧间 ORB 特征 → 部分仿射(RANSAC) → 累积到参考帧，消除相机运动。
    // 稳像默认关：当前这版是"帧间 ORB + 累积仿射"，在有运动目标+手持时会有漂移，
    // 效果反而更差（见 README）。需要时用 SPIN_DEMO_STAB=1 打开做实验。
    const bool stabilize = std::getenv("SPIN_DEMO_STAB") != nullptr &&
                           std::string(std::getenv("SPIN_DEMO_STAB")) != "0";
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1200);
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    cv::Mat T_ref = cv::Mat::eye(2, 3, CV_64F);   // 当前帧坐标 -> 参考帧坐标
    cv::Mat prev_gray, prev_desc;
    std::vector<cv::KeyPoint> prev_kps;
    bool have_prev = false;
    cv::Point2f prev_ctr;
    double t = 0.0;
    int frame_idx = 0;
    int written = 0;
    int lost = 0;

    cv::Mat frame;
    cv::Mat trail = cv::Mat::zeros(H, W, CV_8UC3);   // 累积的"扫掠轨迹"
    while (cap.read(frame) && written < max_frames)
    {
        const double dt = 1.0 / fps;

        cv::Mat aligned;
        if (stabilize)
        {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::KeyPoint> kps;
            cv::Mat desc;
            orb->detectAndCompute(gray, cv::noArray(), kps, desc);
            if (!prev_desc.empty() && !desc.empty())
            {
                std::vector<cv::DMatch> matches;
                matcher.match(prev_desc, desc, matches);
                if (matches.size() >= 12)
                {
                    std::vector<cv::Point2f> p0, p1;
                    for (const auto& m : matches)
                    {
                        p0.push_back(prev_kps[m.queryIdx].pt);
                        p1.push_back(kps[m.trainIdx].pt);
                    }
                    cv::Mat A = cv::estimateAffinePartial2D(p0, p1, cv::noArray(),
                                                            cv::RANSAC, 3.0);  // prev -> cur
                    if (!A.empty())
                    {
                        cv::Mat A32;
                        A.convertTo(A32, CV_32F);
                        cv::Mat A32_inv;
                        cv::invertAffineTransform(A32, A32_inv);   // cur -> prev
                        cv::Mat Ainv64;
                        A32_inv.convertTo(Ainv64, CV_64F);
                        cv::Mat Tprev64 = T_ref.clone();
                        cv::Mat A3 = cv::Mat::eye(3, 3, CV_64F);
                        Ainv64.copyTo(A3(cv::Rect(0, 0, 3, 2)));
                        cv::Mat T3 = cv::Mat::eye(3, 3, CV_64F);
                        Tprev64.copyTo(T3(cv::Rect(0, 0, 3, 2)));
                        cv::Mat Tcur3 = T3 * A3;
                        T_ref = Tcur3(cv::Rect(0, 0, 3, 2)).clone();
                    }
                }
            }
            cv::warpAffine(frame, aligned, T_ref, cv::Size(W, H), cv::INTER_LINEAR,
                           cv::BORDER_REPLICATE);
            prev_gray = gray;
            prev_kps = kps;
            prev_desc = desc;
        }
        else
        {
            aligned = frame.clone();
        }

        const auto dets = detector.detect(aligned);

        // 选一块板连续跟踪（离上一帧位置最近；首帧选置信度最高）
        const auto_aim::NnDetection* best = nullptr;
        {
            double best_score = -1e18;
            const double jump = 0.35 * std::min(W, H);   // 连续跟踪的最大帧间跳变
            for (const auto& d : dets)
            {
                if (d.keypoints.size() < 4) continue;
                const cv::Point2f ctr = 0.25f * (d.keypoints[0] + d.keypoints[1] +
                                                 d.keypoints[2] + d.keypoints[3]);
                if (have_prev)
                {
                    const double dist = std::hypot(ctr.x - prev_ctr.x, ctr.y - prev_ctr.y);
                    if (dist > jump) continue;          // 跳太远 = 换板，不跟
                    if (-dist > best_score) { best_score = -dist; best = &d; }
                }
                else if (d.confidence > best_score) { best_score = d.confidence; best = &d; }
            }
        }

        cv::Mat vis = aligned.clone();
        if (best)
        {
            // 关键点顺序：0=left.top, 1=right.top, 2=right.bottom, 3=left.bottom
            const cv::Point2f lt = best->keypoints[0], rt = best->keypoints[1];
            const cv::Point2f rb = best->keypoints[2], lb = best->keypoints[3];
            const bool left_first = (lt.x + lb.x) < (rt.x + rb.x);
            const int left_side = -1, right_side = 1;
            obs.push_back(makeObs(t, 0, left_first ? left_side : right_side, lt, lb));
            obs.push_back(makeObs(t, 0, left_first ? right_side : left_side, rt, rb));

            cv::line(vis, lt, lb, cv::Scalar(0, 255, 0), 2);
            cv::line(vis, rt, rb, cv::Scalar(0, 255, 0), 2);
            bars_csv << t << ',' << lt.x << ',' << lt.y << ',' << lb.x << ',' << lb.y << ','
                     << rt.x << ',' << rt.y << ',' << rb.x << ',' << rb.y << '\n';
            for (const auto& kp : best->keypoints) cv::circle(vis, kp, 3, cv::Scalar(0, 255, 255), -1);
            // 累积轨迹：把这两根灯条线段画到 trail 上（浅色），形成"扫过的轨迹带"
            cv::line(trail, lt, lb, cv::Scalar(90, 60, 0), 2);
            cv::line(trail, rt, rb, cv::Scalar(90, 60, 0), 2);
            prev_ctr = 0.25f * (lt + rt + rb + lb);
            have_prev = true;
            lost = 0;
        }
        else
        {
            ++lost;
            if (lost > 15)   // 连续丢太久：重置，重新捕获（避免把另一块板混进同一段）
            {
                obs.clear();
                have_prev = false;
                last = auto_aim::SpinGeometryResult{};
            }
        }

        // 每 10 帧（且有足够数据）重拟合一次
        if (obs.size() >= 20 && frame_idx % 10 == 0)
        {
            auto_aim::SpinGeometryEstimator est(gcfg);
            if (est.fit(obs)) last = est.result();
        }

        if (last.valid && last.spline.size() >= 4)
        {
            // 叠加：拟合预测的该板两根灯条（青）+ 旋转中心投影（黄）+ 半径圆
            Eigen::Vector3d top, bot;
            // 复用估计器内部几何：直接用 allBars 的第 0 块板的 ±1 —— 通过重新算
            const double th = last.yawAt(t);
            const Eigen::Vector3d axis(last.axis_tx, 1.0, last.axis_tz);
            const Eigen::Vector3d a = axis.normalized();
            Eigen::Vector3d ref(1, 0, 0);
            if (std::abs(a.dot(ref)) > 0.9) ref = Eigen::Vector3d(0, 1, 0);
            const Eigen::Vector3d e1 = (ref - a * ref.dot(a)).normalized();
            const Eigen::Vector3d e2 = e1.cross(a);
            for (int side = -1; side <= 1; side += 2)
            {
                const double ct = std::cos(th), st = std::sin(th);
                const Eigen::Vector3d rhat = ct * e1 + st * e2;
                const Eigen::Vector3d that = -st * e1 + ct * e2;
                const Eigen::Vector3d bc = last.center + rhat * last.radius
                    + that * (side * c_off);
                const Eigen::Vector3d dir = std::cos(last.pitch) * a + std::sin(last.pitch) * rhat;
                top = bc + dir * (0.5 * L_len);
                bot = bc - dir * (0.5 * L_len);
                auto prj = [&](const Eigen::Vector3d& p) {
                    return cv::Point2f(static_cast<float>(fx * p.x() / p.z() + cx),
                                       static_cast<float>(fx * p.y() / p.z() + cy));
                };
                if (top.z() > 0.2 && bot.z() > 0.2)
                    cv::line(vis, prj(top), prj(bot), cv::Scalar(255, 255, 0), 2);
            }
            const Eigen::Vector3d Cp = last.center;
            if (Cp.z() > 0.2)
                cv::drawMarker(vis, cv::Point2f(static_cast<float>(fx * Cp.x() / Cp.z() + cx),
                                                static_cast<float>(fx * Cp.y() / Cp.z() + cy)),
                               cv::Scalar(0, 200, 255), cv::MARKER_CROSS, 24, 2);
            cv::putText(vis,
                        cv::format("R=%.3fm pitch=%.1fdeg arc=%.0fdeg reproj=%.1fpx",
                                   last.radius, last.pitch * 180.0 / M_PI,
                                   last.arc_coverage_deg, last.reproj_rmse_px),
                        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            cv::putText(vis, cv::format("yaw=%.1fdeg  obs=%d", last.yawAt(t) * 180.0 / M_PI,
                                        static_cast<int>(obs.size())),
                        cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
        cv::putText(vis, "green=detected bars  cyan=fitted bars  yellow=center",
                    cv::Point(20, H - 24), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(200, 255, 200), 2, cv::LINE_AA);

        cv::addWeighted(vis, 1.0, trail, 0.9, 0.0, vis);   // 叠加累积轨迹
        writer.write(vis);
        ++written;
        ++frame_idx;
        t += dt;
    }
    std::cout << "wrote " << out_path << "  frames=" << written
              << "  obs=" << obs.size()
              << "  valid=" << (last.valid ? 1 : 0)
              << (last.valid ? cv::format("  R=%.3f pitch=%.1fdeg arc=%.0fdeg reproj=%.1fpx",
                                          last.radius, last.pitch * 180.0 / M_PI,
                                          last.arc_coverage_deg, last.reproj_rmse_px) : "")
              << "\n";
    return 0;
}
