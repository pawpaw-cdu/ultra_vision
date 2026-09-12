#include "../include/utils.hpp"
#include <cmath>

namespace energy {

float normalizeAngle(float angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

void showBladesROI(const cv::Mat& src, const std::vector<cv::Point2f>& blade_centers, int roi_size) {
    if (blade_centers.empty()) {
        cv::Mat empty(80, 80, CV_8UC3, cv::Scalar(0,0,0));
        cv::imshow("Detected Blades ROI", empty);
        return;
    }
    int roi_w = roi_size, roi_h = roi_size;
    int cols = std::min(3, (int)blade_centers.size());
    int rows = (blade_centers.size() + cols - 1) / cols;
    cv::Mat canvas(rows * roi_h, cols * roi_w, src.type(), cv::Scalar(255,255,255));
    for (size_t i = 0; i < blade_centers.size(); ++i) {
        int x = (int)blade_centers[i].x;
        int y = (int)blade_centers[i].y;
        int half = roi_size / 2;
        cv::Rect roi_rect(std::max(0, x - half), std::max(0, y - half), roi_size, roi_size);
        roi_rect &= cv::Rect(0, 0, src.cols, src.rows);
        if (roi_rect.width <= 0 || roi_rect.height <= 0) continue;
        cv::Mat roi_img = src(roi_rect);
        cv::Mat resized;
        if (roi_img.size() != cv::Size(roi_w, roi_h))
            cv::resize(roi_img, resized, cv::Size(roi_w, roi_h));
        else
            resized = roi_img.clone();
        cv::Point2f center_in_roi(roi_w/2.0f, roi_h/2.0f);
        cv::circle(resized, center_in_roi, 3, cv::Scalar(0,0,255), 1);
        cv::line(resized, cv::Point(center_in_roi.x-5, center_in_roi.y), cv::Point(center_in_roi.x+5, center_in_roi.y), cv::Scalar(0,255,0), 1);
        cv::line(resized, cv::Point(center_in_roi.x, center_in_roi.y-5), cv::Point(center_in_roi.x, center_in_roi.y+5), cv::Scalar(0,255,0), 1);
        int row = i / cols;
        int col = i % cols;
        cv::Rect canvas_rect(col * roi_w, row * roi_h, roi_w, roi_h);
        resized.copyTo(canvas(canvas_rect));
    }
    cv::imshow("Detected Blades ROI", canvas);
}

void drawModel(cv::Mat& img, const TrackerResult& result) {
    if (!result.valid) {
        cv::putText(img, "Model NOT initialized", cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        return;
    }
    cv::circle(img, result.model_center, 5, cv::Scalar(0, 255, 0), -1);
    //cv::circle(img, result.model_center, result.model_radius, cv::Scalar(0, 255, 0), 2);
    cv::putText(img, "Model Center: (" + std::to_string((int)result.model_center.x) + "," + std::to_string((int)result.model_center.y) + ")",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::putText(img, "Model Radius: " + std::to_string((int)result.model_radius),
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::putText(img, "Model OK", cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    for (const auto& blade : result.blades) {
        for (int i = 0; i < 4; ++i) {
            cv::line(img, blade.object_points[i], blade.object_points[(i+1)%4], cv::Scalar(255, 255, 255), 2);
        }
        for (int i = 0; i < 4; ++i){
            cv::line(img, blade.transforme_points[i], blade.transforme_points[(i+1)%4], cv::Scalar(255, 255, 255), 2);
        }
        for (int i = 0; i < 4; ++i){
            cv::line(img, blade.transforme_points[i], blade.object_points[i], cv::Scalar(255, 255, 255), 2);
        }
        cv::circle(img, blade.center, 5, cv::Scalar(0, 255, 0), -1);
        cv::line(img, result.model_center, blade.center, cv::Scalar(255, 100, 100), 1);
    }
}

} // namespace energy
