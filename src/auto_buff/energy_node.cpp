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

enum isActive {ACTIVE, NOACTIVE, NOTHING};

struct Blade{
    vector<Point2f> centers;
    vector<isActive> blade_status{5, NOTHING};
};

struct EnergyBlade {
    cv::Point2f center;
    int id;
    float angle_rad;
    cv::Point2f object_points[4];
};

struct EnergyDetectorConfig {
    int color = 1;    // red = 1, blue = 0
    float angular_velocity_deg = 60.0f;
    int num_blades = 5;
    float jump_threshold_px = 20.0f;
    float min_radius_px = 30.0f;
    float max_radius_px = 350.0f;
    int max_track_lost_frames = 5;
    float max_match_distance_ratio = 0.4f;
};

// ==================== 基于HSV的扇叶中心检测 ====================

void detectBladeCentersHSV(const Mat& bgr, Blade& blade, int color) {
    //vector<Point2f> centers;
    if (bgr.empty()) {
        std::cerr <<"[BladeDetect]: Img or Video wrong"<<std::endl;
    }

    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Mat mask;
    Mat mask1, mask2;
    int n = 1;
    if(color){
        // 红色范围
    inRange(hsv, Scalar(0, 80, 80), Scalar(60, 255, 255), mask1);
    //inRange(hsv, Scalar(140, 105, 105), Scalar(180, 252, 253), mask2);
    mask = mask1;
    n = 3;
    }else{
        // blue
    inRange(hsv, Scalar(88, 68, 198), Scalar(120, 255, 255), mask1);
    inRange(hsv, Scalar(55, 0, 200), Scalar(130, 45, 255), mask2);
    mask = mask1 | mask2;
    n = 5;
    }

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(n, n));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    // 可选显示掩膜（调试）
    Mat mask_disp;
    resize(mask, mask_disp, Size(960,540));
    imshow("mask_blade", mask_disp);

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
    //     std::cout <<i<<"] center: "<<rect.center<<std::endl;
    //     std::cout <<i<<"] area: "<<contourArea(contour)<<std::endl;
    //     std::cout <<i<<"] rect_area: "<<(rect.size.height * rect.size.width)<<std::endl;
    //     std::cout <<i<<"] position: "<< rect.center<<std::endl;
    //     std::cout <<i<<"] height-width: "<<rect.size.height <<" "<<rect.size.width << std::endl;
    //     i++;
    //             }
    // cv::resize(bgr_clone,bgr_clone,Size(960, 540));
    // imshow("bgr_debug", bgr_clone);

    int k = 0;
    for (const auto& cnt : contours) {
        isActive status;
        //k++;
        double area = contourArea(cnt);
        if (area > 7000) {
            if(area < 11000){
                status = NOACTIVE;
            }else if(area < 15000 && area > 13000){
                status = ACTIVE;
            }else{
                continue;
            }
        }else{
            continue;
        }
        RotatedRect rect = minAreaRect(cnt);
        float rect_area = rect.size.width * rect.size.height;
        if (rect_area <= 0) continue;
        double rectangularity = area / rect_area;
        //std::cout <<k<<"] rectangularity: "<<rectangularity<<std::endl;
        if (rectangularity < 0.2) continue;

        float w = rect.size.width;
        float h = rect.size.height;
        float ratio = min(w, h) / max(w, h);
        //std::cout <<k<<"] ratio: "<<ratio<<std::endl;
        if (ratio > 0.75) continue;  // 细长

        double perimeter = arcLength(cnt, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        //std::cout <<k<<"] circularity: "<<circularity<<std::endl;
        if (circularity > 0.7) continue;

        vector<Point> hull;
        convexHull(cnt, hull);
        double hull_area = contourArea(hull);
        double solidity = area / hull_area;
        //std::cout <<k<<"] solidity: "<<solidity<<std::endl;
        if (solidity < 0.5) continue;

        Moments m = moments(cnt);
        if (m.m00 > 0) {
            blade.centers.push_back(Point2f(m.m10 / m.m00, m.m01 / m.m00));
            //blade.centers.at((k % 5)) = Point2f(m.m10 / m.m00, m.m01 / m.m00);
            //blade.blade_status.at((k % 5)) = status;
            //k++;
        }
    }
}

// ==================== 基于HSV的R标（中心圆）检测 ====================

bool detectRCenterHSV(const Mat& bgr, Point2f& center, float& radius, float min_radius, float max_radius, int color) {
    if (bgr.empty()) {
        std::cout <<"bgr nothing!"<<std::endl;
        return false;
    }
    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    Mat mask;
    Mat mask1, mask2;
    int n = 1;
    if(color){
        // 红色范围
    inRange(hsv, Scalar(5, 110, 115), Scalar(33, 255, 255), mask1);
    inRange(hsv, Scalar(155, 110, 115), Scalar(180, 255, 255), mask2);
    mask = mask1 | mask2;
    n = 5;
    }else{
        // blue
    inRange(hsv, Scalar(88, 68, 198), Scalar(120, 255, 255), mask1);
    inRange(hsv, Scalar(55, 0, 200), Scalar(130, 45, 255), mask2);
    mask = mask1 | mask2;
    n = 3;
    }

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(n, n));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    

    // Mat mask_disp;
    // resize(mask, mask_disp, Size(960,540));
    // imshow("mask_R", mask_disp);

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
        if (area < 200 || area > 400) {
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
    string video_path = "/home/pawpaw-ubuntu/rm_ultra/test_vid_pic/buff_red.mp4";
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open video file " << video_path << endl;
        return -1;
    }

    EnergyDetectorConfig cfg;  // 原配置
    cfg.color = 1;
    cfg.num_blades = 5;
    cfg.angular_velocity_deg = 50.5f;
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

    // 新增：每个扇叶独立平滑角度
    vector<float> blade_smoothed_angles(cfg.num_blades, 0.0f);
    vector<bool> blade_angle_initialized(cfg.num_blades, false);
    const float BLADE_ANGLE_SMOOTH = 0.65f;   // 扇叶角度平滑系数

    // ========== 新增：扇叶跟踪器（用于匈牙利匹配和丢失预测）==========
    struct BladeTracker {
        int id;
        Point2f current_center;
        float current_angle;
        bool lost;
        int lost_count;
        Point2f predicted_center;
        float predicted_angle;
    };
    vector<BladeTracker> blade_trackers(cfg.num_blades);
    for (int i = 0; i < cfg.num_blades; ++i) {
        blade_trackers[i].id = i;
        blade_trackers[i].lost = true;
        blade_trackers[i].lost_count = 0;
        blade_trackers[i].current_center = Point2f(0,0);
        blade_trackers[i].current_angle = 0;
        blade_trackers[i].predicted_center = Point2f(0,0);
        blade_trackers[i].predicted_angle = 0;
    }

    // 匈牙利算法（最小权匹配）简化实现，用于5xN的分配（N≤5）
    auto hungarian = [](const vector<vector<float>>& cost) -> vector<int> {
        int n = cost.size();      // 检测到的扇叶数
        int m = cost[0].size();   // 跟踪器数（5）
        // 贪心匹配：按最小代价顺序分配（简单实现，对于小规模足够）
        vector<int> assignment(m, -1);
        vector<bool> used(n, false);
        for (int iter = 0; iter < m; ++iter) {
            int best_i = -1, best_j = -1;
            float best_cost = 1e9;
            for (int j = 0; j < m; ++j) {
                if (assignment[j] != -1) continue;
                for (int i = 0; i < n; ++i) {
                    if (used[i]) continue;
                    if (cost[i][j] < best_cost) {
                        best_cost = cost[i][j];
                        best_i = i;
                        best_j = j;
                    }
                }
            }
            if (best_i != -1) {
                assignment[best_j] = best_i;
                used[best_i] = true;
            }
        }
        return assignment;
    };

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
        Blade blade;
        detectBladeCentersHSV(frame, blade, cfg.color);
        //vector<Point2f> blade_centers = blade.centers;
        Point2f r_center;
        float r_radius;
        bool r_found = detectRCenterHSV(frame, r_center, r_radius, 10, 35, cfg.color);

        // 2. 模型构建：圆心直接使用R标检测结果
        Point2f model_center = r_center;
        bool model_valid = r_found;

        // 计算半径：若有扇叶点，则用圆心到扇叶点的平均距离作为半径；否则使用R标本身的半径
        float model_radius = 0;
        int valid_blade_count = 0;
        // 先使用跟踪器中未丢失的扇叶中心进行半径估计（优先）
        vector<Point2f> valid_centers;
        for (auto& tr : blade_trackers) {
            if (!tr.lost && tr.current_center.x != 0 && tr.current_center.y != 0) {
                valid_centers.push_back(tr.current_center);
            }
        }
        if (model_valid && !valid_centers.empty()) {
            float sum_dist = 0;
            for (const auto& pt : valid_centers) {
                sum_dist += norm(pt - model_center);
            }
            model_radius = sum_dist / valid_centers.size();
            model_radius *= 1.1;
            temp_radius = model_radius;
            valid_blade_count = valid_centers.size();
        } else if (model_valid && !blade.centers.empty()) {
            // 如果跟踪器无有效点，直接使用当前检测的扇叶中心
            float sum_dist = 0;
            for (const auto& blade_pt : blade.centers) {
                sum_dist += norm(blade_pt - model_center);
            }
            model_radius = sum_dist / blade.centers.size();
            temp_radius = model_radius;
            valid_blade_count = blade.centers.size();
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

        // ========== 核心优化：基于匈牙利算法的数据关联 ==========
        // 准备当前帧检测到的扇叶中心
        vector<Point2f> detected_centers = blade.centers;
        if (!detected_centers.empty()) {
            // 构建代价矩阵（检测数 × 跟踪器数）
            vector<vector<float>> cost(detected_centers.size(), vector<float>(cfg.num_blades, 0));
            for (size_t i = 0; i < detected_centers.size(); ++i) {
                for (int j = 0; j < cfg.num_blades; ++j) {
                    // 代价为欧氏距离，如果跟踪器丢失，增大代价（使其不易匹配）
                    float dist = norm(detected_centers[i] - blade_trackers[j].current_center);
                    cost[i][j] = dist;
                    if (blade_trackers[j].lost) cost[i][j] += 50.0f; // 丢失的跟踪器有额外惩罚
                }
            }
            // 执行匹配
            vector<int> assignment = hungarian(cost); // assignment[j] = 匹配到的检测索引，-1表示未匹配
            // 更新每个跟踪器的状态
            for (int j = 0; j < cfg.num_blades; ++j) {
                int matched_idx = assignment[j];
                if (matched_idx != -1) {
                    // 匹配成功
                    Point2f matched_center = detected_centers[matched_idx];
                    blade_trackers[j].current_center = matched_center;
                    blade_trackers[j].lost = false;
                    blade_trackers[j].lost_count = 0;
                    // 计算角度
                    float dx = matched_center.x - vis_center.x;
                    float dy = matched_center.y - vis_center.y;
                    float measured_angle = atan2(dy, dx);
                    // 角度平滑
                    if (blade_angle_initialized[j]) {
                        float diff = measured_angle - blade_smoothed_angles[j];
                        if (diff > M_PI) diff -= 2*M_PI;
                        if (diff < -M_PI) diff += 2*M_PI;
                        blade_smoothed_angles[j] += BLADE_ANGLE_SMOOTH * diff;
                    } else {
                        blade_smoothed_angles[j] = measured_angle;
                        blade_angle_initialized[j] = true;
                    }
                    blade_trackers[j].current_angle = blade_smoothed_angles[j];
                } else {
                    // 未匹配到，标记为丢失，并增加丢失计数
                    blade_trackers[j].lost = true;
                    blade_trackers[j].lost_count++;
                    // 预测：基于恒定角速度外推（可选）
                    if (blade_angle_initialized[j]) {
                        float angular_vel_rad = cfg.angular_velocity_deg * M_PI / 180.0f;
                        blade_smoothed_angles[j] += angular_vel_rad * dt;
                        // 规范化
                        while (blade_smoothed_angles[j] > M_PI) blade_smoothed_angles[j] -= 2*M_PI;
                        while (blade_smoothed_angles[j] < -M_PI) blade_smoothed_angles[j] += 2*M_PI;
                        // 预测位置（用于后续半径计算或显示）
                        float pred_angle = blade_smoothed_angles[j];
                        blade_trackers[j].predicted_center = Point2f(vis_center.x + vis_radius * cos(pred_angle),
                                                                      vis_center.y + vis_radius * sin(pred_angle));
                        // 若丢失帧数超过阈值，重置该跟踪器
                        if (blade_trackers[j].lost_count > cfg.max_track_lost_frames) {
                            blade_angle_initialized[j] = false;
                            blade_trackers[j].lost_count = 0;
                        }
                    }
                }
            }
        } else {
            // 没有检测到任何扇叶：所有跟踪器标记为丢失，并进行运动预测
            float angular_vel_rad = cfg.angular_velocity_deg * M_PI / 180.0f;
            for (int j = 0; j < cfg.num_blades; ++j) {
                if (blade_angle_initialized[j]) {
                    blade_smoothed_angles[j] += angular_vel_rad * dt;
                    while (blade_smoothed_angles[j] > M_PI) blade_smoothed_angles[j] -= 2*M_PI;
                    while (blade_smoothed_angles[j] < -M_PI) blade_smoothed_angles[j] += 2*M_PI;
                    blade_trackers[j].predicted_center = Point2f(vis_center.x + vis_radius * cos(blade_smoothed_angles[j]),
                                                                  vis_center.y + vis_radius * sin(blade_smoothed_angles[j]));
                }
                blade_trackers[j].lost = true;
                blade_trackers[j].lost_count++;
                if (blade_trackers[j].lost_count > cfg.max_track_lost_frames) {
                    blade_angle_initialized[j] = false;
                    blade_trackers[j].lost_count = 0;
                }
            }
        }

        // 更新参考角度标志（至少有一个跟踪器未丢失或已初始化）
        bool have_active_tracker = false;
        for (int j = 0; j < cfg.num_blades; ++j) {
            if (!blade_trackers[j].lost && blade_angle_initialized[j]) {
                have_active_tracker = true;
                break;
            }
        }
        if (have_active_tracker) {
            has_ref_angle = true;
            // 取第一个有效跟踪器的角度作为参考（用于无检测时的外推）
            for (int j = 0; j < cfg.num_blades; ++j) {
                if (!blade_trackers[j].lost && blade_angle_initialized[j]) {
                    last_ref_angle = blade_smoothed_angles[j];
                    predicted_ref_angle = last_ref_angle;
                    break;
                }
            }
        } else if (has_ref_angle) {
            // 没有任何跟踪器活动，但之前有历史，则继续角速度外推
            float angular_vel_rad = cfg.angular_velocity_deg * M_PI / 180.0f;
            predicted_ref_angle += angular_vel_rad * dt;
            if (predicted_ref_angle > M_PI) predicted_ref_angle -= 2*M_PI;
            if (predicted_ref_angle < -M_PI) predicted_ref_angle += 2*M_PI;
            last_ref_angle = predicted_ref_angle;
        } else {
            has_ref_angle = false;
        }

        // 5. 可视化
        Mat display = frame.clone();

        // 绘制所有检测到的扇叶中心（Blue）
        for (const auto& pt : blade.centers) {
            circle(display, pt, 5, Scalar(255, 0, 0), -1);
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

            // 预测并绘制完整的五个扇叶（使用跟踪器的平滑角度）
            for (int i = 0; i < cfg.num_blades; ++i) {
                float angle;
                if (blade_angle_initialized[i]) {
                    angle = blade_smoothed_angles[i];
                } else if (has_ref_angle && i > 0 && blade_angle_initialized[0]) {
                    // 未初始化的扇叶根据第一个扇叶等间隔推算
                    float step = 2 * M_PI / cfg.num_blades;
                    angle = blade_smoothed_angles[0] + i * step;
                } else {
                    continue;
                }
                Point2f predicted_blade(vis_center.x + vis_radius * cos(angle),
                                        vis_center.y + vis_radius * sin(angle));

                // 绘制扇叶主体（四个边角点，模拟一个十字形或矩形）
                Point2f blade_points[4];
                Point2f blade_to_points[4];
                float blade_size = 50.0f;
                blade_points[0] = Point2f(predicted_blade.x + 10.0f, predicted_blade.y - blade_size );
                blade_points[1] = Point2f(predicted_blade.x - blade_size + 10.0f, predicted_blade.y );
                blade_points[2] = Point2f(predicted_blade.x + 10.0f, predicted_blade.y + blade_size );
                blade_points[3] = Point2f(predicted_blade.x + blade_size + 10.0f, predicted_blade.y );
                
                blade_to_points[0] = Point2f(predicted_blade.x - 10.0f, predicted_blade.y - blade_size);
                blade_to_points[1] = Point2f(predicted_blade.x - blade_size - 10.0f, predicted_blade.y);
                blade_to_points[2] = Point2f(predicted_blade.x - 10.0f, predicted_blade.y + blade_size);
                blade_to_points[3] = Point2f(predicted_blade.x + blade_size - 10.0f, predicted_blade.y);

                for (int j = 0; j < 4; ++j) {
                    line(display, blade_points[j], blade_points[(j+1)%4], Scalar(255, 255, 255), 2);
                }
                for (int j = 0; j < 4; ++j) {
                    line(display, blade_to_points[j], blade_to_points[(j+1)%4], Scalar(255, 255, 255), 2);
                }
                for (int j = 0; j < 4; ++j) {
                    line(display, blade_points[j], blade_to_points[j], Scalar(255, 255, 255), 2);
                }
                
                // 绘制扇叶中心点（绿色）
                circle(display, predicted_blade, 5, Scalar(0, 255, 0), -1);
                // 连线到圆心
                line(display, vis_center, predicted_blade, Scalar(255, 100, 100), 1);
            }
        } else if (model_valid && vis_radius <= 0 && !blade.centers.empty()) {
            // 半径未知时，仅用检测到的扇叶连线
            for (const auto& blade : blade.centers) {
                line(display, vis_center, blade, Scalar(0, 255, 255), 1);
            }
        }

        // 文字提示
        putText(display, "Energy Detector (Autonomous Motion)", Point(30, 30),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        putText(display, "Blades detected: " + to_string(blade.centers.size()), Point(30, 60),
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

        showBladesROI(frame, blade.centers, 80);

        if (waitKey(10) == 'q') break;
    }
    destroyAllWindows();
    return 0;
}
