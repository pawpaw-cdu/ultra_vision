#include "visualizer.h"

#include <cmath>

#ifdef RM_HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace rm
{
    void visualizeResults(const std::vector<SimulationFrame>& frames,
                          const std::optional<FitResult>& fit,
                          const CameraIntrinsics& K, double gimbal_pitch, double alpha, double R,
                          double L, const std::string& output_path)
    {
#ifdef RM_HAVE_OPENCV
        const int W = static_cast<int>(2 * K.cx), H = static_cast<int>(2 * K.cy);
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
        std::vector<cv::Point> fit_top, fit_bot;
        for (const auto& f : frames)
        {
            cv::circle(img, cv::Point(cvRound(f.top_px.x()), cvRound(f.top_px.y())), 2,
                       cv::Scalar(200, 120, 0), -1);
            cv::circle(img, cv::Point(cvRound(f.bottom_px.x()), cvRound(f.bottom_px.y())), 2,
                       cv::Scalar(40, 40, 220), -1);
            if (fit)
            {
                Vec3 top, bot;
                barEndpointsCamera(fit->centerAt(f.timestamp, frames.front().timestamp), fit->h,
                                   fit->yawAt(f.timestamp), alpha, R, L, gimbal_pitch, top, bot);
                const Vec2 pt = project(top, K), pb = project(bot, K);
                fit_top.emplace_back(cvRound(pt.x()), cvRound(pt.y()));
                fit_bot.emplace_back(cvRound(pb.x()), cvRound(pb.y()));
            }
        }
        if (fit)
        {
            cv::polylines(img, fit_top, false, cv::Scalar(0, 200, 0), 2, cv::LINE_AA);
            cv::polylines(img, fit_bot, false, cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
        }
        cv::putText(img, "obs top=blue bot=red | fit green/orange", cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(30, 30, 30), 2);
        cv::imwrite(output_path, img);
#else
        (void)frames; (void)fit; (void)K; (void)gimbal_pitch; (void)alpha; (void)R; (void)L;
        (void)output_path;
#endif
    }
}  // namespace rm
