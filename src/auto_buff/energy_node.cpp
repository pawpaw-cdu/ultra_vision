#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;


const float CENTER_SMOOTH = 0.9f;   // 圆心平滑系数 (0~1)，1 表示无平滑，0.3 表示较强平滑
const float RADIUS_SMOOTH = 0.9f;   // 半径平滑系数 (0~1)
const float ANGLE_SMOOTH = 0.75f;    // 角度平滑系数，用于检测到扇叶后快速融合预测值

// ========================== 能量机关检测器定义 ==========================

struct EnergyBlade {
    cv::Point2f center;
    int id;
    float angle_rad;
    cv::Point2f object_points[4];
};

struct EnergyDetectorConfig {
    float angular_velocity_deg = 60.0f;
    int num_blades = 5;
    float jump_threshold_px = 20.0f;
    float min_radius_px = 30.0f;
    float max_radius_px = 350.0f;
    int max_track_lost_frames = 5;
    float max_match_distance_ratio = 0.4f;
};

// ==================== 基于HSV的扇叶中心检测 ====================

vector<Point2f> detectBladeCentersHSV(const Mat& bgr) {
    vector<Point2f> centers;
    if (bgr.empty()) return centers;

    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Mat mask;
    Mat mask1, mask2;
    // 红色范围
    inRange(hsv, Scalar(0, 80, 80), Scalar(60, 255, 255), mask1);
    inRange(hsv, Scalar(120, 80, 80), Scalar(180, 255, 255), mask2);
    mask = mask1 | mask2;

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    // 可选显示掩膜（调试）
    // Mat mask_disp;
    // resize(mask, mask_disp, Size(960,540));
    // imshow("mask", mask_disp);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);
        if (area < 8000 || area > 50000) continue;  // 根据实际情况调整

        RotatedRect rect = minAreaRect(cnt);
        float rect_area = rect.size.width * rect.size.height;
        if (rect_area <= 0) continue;
        double rectangularity = area / rect_area;
        if (rectangularity < 0.5) continue;

        float w = rect.size.width;
        float h = rect.size.height;
        float ratio = min(w, h) / max(w, h);
        if (ratio > 0.65) continue;  // 细长

        double perimeter = arcLength(cnt, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity > 0.5) continue;

        vector<Point> hull;
        convexHull(cnt, hull);
        double hull_area = contourArea(hull);
        double solidity = area / hull_area;
        if (solidity < 0.5) continue;

        Moments m = moments(cnt);
        if (m.m00 > 0) {
            centers.push_back(Point2f(m.m10 / m.m00, m.m01 / m.m00));
        }
    }
    return centers;
}

// ==================== 基于HSV的R标（中心圆）检测 ====================

bool detectRCenterHSV(const Mat& bgr, Point2f& center, float& radius, float min_radius, float max_radius) {
    if (bgr.empty()) {
        std::cout <<"bgr nothing!"<<std::endl;
        return false;
    }
    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Mat mask;
    Mat mask1, mask2;
    // 提取红色区域（与扇叶相同颜色范围）
    inRange(hsv, Scalar(0, 80, 80), Scalar(30, 255, 255), mask1);
    inRange(hsv, Scalar(150, 80, 80), Scalar(180, 255, 255), mask2);
    mask = mask1 | mask2;

    // 形态学去除噪点
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    // Mat mask_disp;
    // resize(mask, mask_disp, Size(960,540));
    // imshow("mask", mask_disp);

    vector<vector<Point>> contours;
    
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // int i = 1;
    // Mat bgr_clone = bgr;
    //     for(auto contour : contours){
    //     cv::RotatedRect rect = cv::minAreaRect(contour);
    //     cv::circle(bgr_clone,rect.center,2,cv::Scalar(0,255,0));
    //     cv::putText(bgr_clone,std::to_string(i),rect.center + cv::Point2f(10, -10),
    //                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    //     std::cout <<i<<"] size: "<<contour.size()<<std::endl;
    //     std::cout <<i<<"] area: "<<contourArea(contour)<<std::endl;
    //     std::cout <<i<<"] position: "<< rect.center<<std::endl;
    //     std::cout <<i<<"] height-width: "<<rect.size.height <<" "<<rect.size.width << std::endl;
    //     i++;
    //             }
    // cv::resize(bgr_clone,bgr_clone,Size(960, 540));
    // imshow("bgr_debug", bgr_clone);

    // 筛选圆形轮廓（近似圆形）
    //int j = 0;
    for (const auto& cnt : contours) {
        //j++;
        double area = contourArea(cnt);
        if (area < 1100 || area > 1300) {
            //std::cout <<"one!"<<std::endl;
            continue;  // 中心圆面积范围
        }
        RotatedRect rect = minAreaRect(cnt);
        float w = rect.size.width;
        float h = rect.size.height;
        float ratio = min(w, h) / max(w, h);
        if (ratio < 0.70) {
            //std::cout <<"two!"<<std::endl;
            continue;   // 应接近圆形
        }
        double perimeter = arcLength(cnt, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity < 0.30) {
            //std::cout <<"three!"<<std::endl;
            continue;  // 圆形度要求较高
        }
        // 计算圆心和半径（使用最小外接圆或质心+平均半径）
        Point2f cnt_center;
        float cnt_radius;
        minEnclosingCircle(cnt, cnt_center, cnt_radius);
        //td::cout <<"[Debug] cnt_radius: "<<cnt_radius <<std::endl;
        if (cnt_radius >= min_radius && cnt_radius <= max_radius) {
            //std::cout <<"success! "<<std::endl;
            center = cnt_center;
            radius = cnt_radius;
            return true;
        }
    }
    return false;
}

void showBladesROI(const Mat& src, const vector<Point2f>& blade_centers, int roi_size = 80) {
    if (blade_centers.empty()) {
        // 清空窗口
        Mat empty(80, 80, CV_8UC3, Scalar(0,0,0));
        imshow("Detected Blades ROI", empty);
        return;
    }
    
    // 每个ROI的大小
    int roi_w = roi_size;
    int roi_h = roi_size;
    
    // 计算需要多少行/列来显示所有扇叶（每行最多3个）
    int cols = min(3, (int)blade_centers.size());
    int rows = (blade_centers.size() + cols - 1) / cols;
    
    // 创建一张白底大图，用于拼接所有ROI
    Mat canvas(rows * roi_h, cols * roi_w, src.type(), Scalar(255,255,255));
    
    for (size_t i = 0; i < blade_centers.size(); ++i) {
        int x = (int)blade_centers[i].x;
        int y = (int)blade_centers[i].y;
        // 提取ROI区域（边界检查）
        int half = roi_size / 2;
        Rect roi_rect(max(0, x - half), max(0, y - half), roi_size, roi_size);
        // 如果超出图像边界，进行裁剪
        roi_rect &= Rect(0, 0, src.cols, src.rows);
        if (roi_rect.width <= 0 || roi_rect.height <= 0) continue;
        
        Mat roi_img = src(roi_rect);
        // 缩放至统一大小（若边界导致尺寸不同，再resize）
        Mat resized;
        if (roi_img.size() != Size(roi_w, roi_h)) {
            resize(roi_img, resized, Size(roi_w, roi_h));
        } else {
            resized = roi_img.clone();
        }
        
        // 在ROI图像上绘制扇叶中心十字标记
        Point2f center_in_roi(roi_w/2.0f, roi_h/2.0f);
        circle(resized, center_in_roi, 3, Scalar(0,0,255), 1);
        line(resized, Point(center_in_roi.x-5, center_in_roi.y), Point(center_in_roi.x+5, center_in_roi.y), Scalar(0,255,0), 1);
        line(resized, Point(center_in_roi.x, center_in_roi.y-5), Point(center_in_roi.x, center_in_roi.y+5), Scalar(0,255,0), 1);
        
        // 计算在画布中的位置
        int row = i / cols;
        int col = i % cols;
        Rect canvas_rect(col * roi_w, row * roi_h, roi_w, roi_h);
        resized.copyTo(canvas(canvas_rect));
    }
    
    imshow("Detected Blades ROI", canvas);
}

// ========================== 主程序 ==========================

int main() {
    string video_path = "/home/pawpaw-ubuntu/rm_ultra/test_vid_pic/buff_three.mp4";
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file " << video_path << endl;
        return -1;
    }

    EnergyDetectorConfig cfg;  // 原配置
    cfg.num_blades = 5;
    cfg.angular_velocity_deg = 55.0f;
    cfg.jump_threshold_px = 10.0f;
    cfg.min_radius_px = 160.0f;
    cfg.max_radius_px = 300.0f;
    cfg.max_track_lost_frames = 5;
    cfg.max_match_distance_ratio = 0.3f;

    // 用于平滑显示的缓存
    Point2f smoothed_center(0, 0);
    float smoothed_radius = 0;
    float temp_radius = 0;
    bool first_frame = true;

    // ===== 自主运动相关变量 =====
    double last_timestamp = 0;          // 上一帧时间戳
    float last_ref_angle = 0;           // 上一次的参考角度（弧度）
    bool has_ref_angle = false;         // 是否已有参考角度
    float predicted_ref_angle = 0;      // 预测的参考角度（用于无检测时）

    namedWindow("Detected Blades ROI", WINDOW_NORMAL);
    resizeWindow("Detected Blades ROI", 320, 240);

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 获取当前时间戳（秒）
        double now = chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
        double dt = 0.0;
        if (last_timestamp > 0) {
            dt = now - last_timestamp;
            if (dt > 0.1) dt = 0.033; // 限制最大间隔
        }
        last_timestamp = now;

        // 1. 实时检测扇叶中心和R标
        vector<Point2f> blade_centers = detectBladeCentersHSV(frame);
        Point2f r_center;
        float r_radius;
        bool r_found = detectRCenterHSV(frame, r_center, r_radius, 10, 35);

        // 2. 模型构建：圆心直接使用R标检测结果
        Point2f model_center = r_center;
        bool model_valid = r_found;

        // 计算半径：若有扇叶点，则用圆心到扇叶点的平均距离作为半径；否则使用R标本身的半径
        float model_radius = 0;
        if (model_valid && !blade_centers.empty()) {
            float sum_dist = 0;
            for (const auto& blade : blade_centers) {
                sum_dist += norm(blade - model_center);
            }
            model_radius = sum_dist / blade_centers.size();
            model_radius *= 1.1;
            temp_radius = model_radius;
        } else if (r_found) {
            model_radius = (temp_radius > 0) ? temp_radius : r_radius;  // 后备：使用R标自身的半径
        }

        // 3. 平滑圆心和半径
        if (first_frame) {
            smoothed_center = model_center;
            smoothed_radius = model_radius;
            first_frame = false;
        } else {
            if (model_valid) {
                smoothed_center = CENTER_SMOOTH * model_center + (1 - CENTER_SMOOTH) * smoothed_center;
                smoothed_radius = RADIUS_SMOOTH * model_radius + (1 - RADIUS_SMOOTH) * smoothed_radius;
            }
        }

        // 用于可视化的圆心和半径
        Point2f vis_center = smoothed_center;
        float vis_radius = smoothed_radius;
        if (!model_valid) {
            // 如果没有检测到R标，则保留上一帧的圆心（平滑值），半径不变
            // vis_center 已经是 smoothed_center，无需额外处理
        }

        // 4. 参考角度的更新（用于扇叶预测）
        float ref_angle = 0;
        bool have_blade = !blade_centers.empty();

        if (have_blade && model_valid) {
            // 有检测到的扇叶时：用第一个扇叶相对于当前圆心的角度作为参考
            float dx = blade_centers[0].x - vis_center.x;
            float dy = blade_centers[0].y - vis_center.y;
            float measured_angle = atan2(dy, dx);
            // 平滑过渡：若已有参考角度，则进行加权融合，避免跳变
            if (has_ref_angle) {
                float angle_diff = measured_angle - last_ref_angle;
                // 规范化角度差到 [-pi, pi]
                if (angle_diff > M_PI) angle_diff -= 2*M_PI;
                if (angle_diff < -M_PI) angle_diff += 2*M_PI;
                measured_angle = last_ref_angle + ANGLE_SMOOTH * angle_diff;
            }
            ref_angle = measured_angle;
            last_ref_angle = ref_angle;
            has_ref_angle = true;
            predicted_ref_angle = ref_angle; // 同步预测值
        } else if (has_ref_angle) {
            // 没有检测到扇叶但有历史角度：根据恒定角速度自行运动
            float angular_vel_rad = cfg.angular_velocity_deg * M_PI / 180.0f;
            predicted_ref_angle += angular_vel_rad * dt;
            // 规范化到 [-pi, pi]
            if (predicted_ref_angle > M_PI) predicted_ref_angle -= 2*M_PI;
            if (predicted_ref_angle < -M_PI) predicted_ref_angle += 2*M_PI;
            ref_angle = predicted_ref_angle;
            last_ref_angle = ref_angle; // 更新参考角度，使下次检测时能平滑融合
        } else {
            // 从未检测到任何扇叶，使用默认角度0
            ref_angle = 0;
        }

        // 5. 可视化
        Mat display = frame.clone();

        // 绘制所有检测到的扇叶中心（红色圆点）
        for (const auto& pt : blade_centers) {
            circle(display, pt, 5, Scalar(0, 0, 255), -1);
        }
        // 绘制R标（蓝色圆心）
        if (r_found) {
            circle(display, r_center, r_radius, Scalar(255, 0, 0), 2);
            circle(display, r_center, 3, Scalar(255, 0, 0), -1);
            putText(display, "R-Center", r_center + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 0), 1);
        }

        // 绘制完整模型（圆心、半径、5个预测扇叶）
        if (vis_radius > 0 && has_ref_angle) {
            // 绘制圆心
            circle(display, vis_center, 5, Scalar(0, 255, 0), -1);
            putText(display, "Model Center", vis_center + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

            // 预测并绘制完整的五个扇叶
            float angle_step = 2 * M_PI / cfg.num_blades;
            for (int i = 0; i < cfg.num_blades; ++i) {
                float angle = ref_angle + i * angle_step;
                Point2f predicted_blade(vis_center.x + vis_radius * cos(angle),
                                        vis_center.y + vis_radius * sin(angle));

                // 绘制扇叶主体（四个边角点，模拟一个十字形或矩形）
                Point2f blade_points[4];
                Point2f blade_to_points[4];
                float blade_size = 90.0f;
                blade_points[0] = Point2f(predicted_blade.x + 10.0f, predicted_blade.y - blade_size );
                blade_points[1] = Point2f(predicted_blade.x - blade_size + 10.0f, predicted_blade.y );
                blade_points[2] = Point2f(predicted_blade.x + 10.0f, predicted_blade.y + blade_size );
                blade_points[3] = Point2f(predicted_blade.x + blade_size + 10.0f, predicted_blade.y );
                
                blade_to_points[0] = Point2f(predicted_blade.x - 10.0f, predicted_blade.y - blade_size);
                blade_to_points[1] = Point2f(predicted_blade.x - blade_size - 10.0f, predicted_blade.y);
                blade_to_points[2] = Point2f(predicted_blade.x - 10.0f, predicted_blade.y + blade_size);
                blade_to_points[3] = Point2f(predicted_blade.x + blade_size - 10.0f, predicted_blade.y);

                for (int j = 0; j < 4; ++j) {
                    line(display, blade_points[j], blade_points[(j+1)%4], Scalar(0, 0, 255), 2);
                }
                for (int j = 0; j < 4; ++j) {
                    line(display, blade_to_points[j], blade_to_points[(j+1)%4], Scalar(0, 0, 255), 2);
                }
                for (int j = 0; j < 4; ++j) {
                    line(display, blade_points[j], blade_to_points[j], Scalar(0, 0, 255), 2);
                }
                // 绘制扇叶中心点（绿色）
                circle(display, predicted_blade, 5, Scalar(0, 255, 0), -1);
                // 连线到圆心
                line(display, vis_center, predicted_blade, Scalar(255, 100, 100), 1);
            }
        } else if (model_valid && vis_radius <= 0 && !blade_centers.empty()) {
            // 半径未知时，仅用检测到的扇叶连线
            for (const auto& blade : blade_centers) {
                line(display, vis_center, blade, Scalar(0, 255, 255), 1);
            }
        }

        // 文字提示
        putText(display, "Energy Detector (Autonomous Motion)", Point(30, 30),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        putText(display, "Blades detected: " + to_string(blade_centers.size()), Point(30, 60),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
        if (r_found) {
            putText(display, "RCenter detected", Point(30, 90),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        }
        if (has_ref_angle) {
            putText(display, "Model radius: " + to_string((int)vis_radius), Point(30, 120),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        }

        resize(display, display, Size(960, 540));
        imshow("Energy Detection", display);

        showBladesROI(frame, blade_centers, 80);

        if (waitKey(12) == 'q') break;
    }
    destroyAllWindows();
    return 0;
}
