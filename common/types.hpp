#ifndef TYPES_HPP
#define TYPES_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace auto_aim
{
    enum ArmorType { small, large };

    struct Light : public cv::RotatedRect
    {
        int color = 0;
        double min_ratio = 0.1;
        double max_ratio = 0.4;
        double max_light_angle = 40.0;

        double tilt_angle;
        double height, width;
        cv::Point2f center, top, bottom;
        cv::RotatedRect rect;

            explicit Light(cv::RotatedRect rect) : cv::RotatedRect(rect)
    {
        cv::Point2f p[4];
        rect.points(p);
        this->center = rect.center;
        float w = rect.size.width;
        float h = rect.size.height;
        float angle_deg = rect.angle;

        // 确定长边、短边和长轴方向角
        float long_axis_rad;
        if (w >= h) {
            this->height = w;
            this->width = h;
            long_axis_rad = angle_deg * CV_PI / 180.0f;
        } else {
            this->height = h;
            this->width = w;
            long_axis_rad = (angle_deg + 90.0f) * CV_PI / 180.0f;
        }

        // 长轴单位向量
        cv::Point2f dir(cos(long_axis_rad), sin(long_axis_rad));
        float half_len = this->height / 2.0f;
        this->top    = this->center - dir * half_len;
        this->bottom = this->center + dir * half_len;

        // 灯条长轴与图像竖直方向的夹角。灯条是一条无向线段，长轴方向
        // (0,-1) 和 (0,1) 必须给出同一个倾角，所以先用 atan2 求方向角，
        // 再折叠到 [-90, 90)。若改用 acos(dir·vertical)，竖直灯条会被算成
        // 0 或 180 度，使 |tilt| < max_angle 的筛选随机丢弃一半竖直灯条，
        // 导致同一块装甲板的两根灯条经常只剩一根、无法配对。
        float tilt_deg = std::atan2(dir.x, -dir.y) * 180.0f / CV_PI;
        if (tilt_deg >= 90.0f) tilt_deg -= 180.0f;
        else if (tilt_deg < -90.0f) tilt_deg += 180.0f;
        this->tilt_angle = tilt_deg;
    
        }

        bool isValid() const {
            float ratio = width / height;
            bool ratio_ok = (min_ratio < ratio && ratio < max_ratio);
            bool angle_ok = (std::abs(tilt_angle) < max_light_angle);
            return ratio_ok && angle_ok;
        }

        Light() = default;
    };

    struct Armor
    {
        Light left, right;
        double center;
        // 板面数字（检测网络 RP24 head 的 num_id：0=sentry…8=Bb；传统灯条流程给 -1）。
        // **必须初始化**：默认构造不赋值时它是 indeterminate，sp_vision 式的
        // "按数字做身份"一旦基于垃圾值就会静默错。
        int number = -1;
        bool solve_result;
        ArmorType armor_type;
        std::vector<cv::Point2f> Points_2D;
        std::vector<cv::Point3f> Points_3D;
        cv::Mat rvec, tvec;

        Armor(const Light& light1, const Light& light2) {
            if (light1.center.x > light2.center.x) {
                left = light2;
                right = light1;
            } else {
                left = light1;
                right = light2;
            }
        }
        Armor() = default;
    };

    // ---------------------------------------------------------------------
    // 局部拟合器（小陀螺 θ/ω + 慢变量 α/R）的数据契约。
    // 见 docs/local_trajectory_fitter.md 与 experiments/local_fitter。
    // ---------------------------------------------------------------------

    /// @brief 一根灯条的上/下端点像素观测（相机系）。
    ///        armor_id 用于多板联合（同一帧可给多块板）；bar_offset 是灯条在板面
    ///        切向的偏移（米，左=+W/2、右=-W/2；提示词单灯条模型取 0）。
    struct LightBarObservation
    {
        double timestamp = 0.0;
        double u_top = 0.0, v_top = 0.0;
        double u_bot = 0.0, v_bot = 0.0;
        int armor_id = 0;
        double bar_offset = 0.0;
        double confidence = 1.0;
    };

    /// @brief 局部拟合器一窗输出。**相机系**（yaw 为绕竖直轴的底盘偏航）。
    ///
    /// `yaw` 的相位约定：它是**底盘 yaw**，且与 `ArmorEKF::angleForPlate()` /
    /// `chassis_model::armorPhaseOffset()`（即 0→+90°、1→0°、2→−90°、3→−180°）
    /// **同一张表**。拟合器自己那张表是 `LocalTrajectoryFitConfig::plate_yaw_offsets`，
    /// 两处必须一致 —— 否则"拟合 yaw"和"EKF yaw"会差 ±90°/180°：四块板完全对称时
    /// 还能靠"模 90° 吸附"掩盖，一旦启用逐板几何（plate_distance / l / h）就会
    /// 让模型分叉。见 docs/local_fit_tracker_responsibility_audit.md 审计项 G。
    struct LocalFitResult
    {
        bool valid = false;
        double timestamp = 0.0;         // 窗口末端时间（拟合历经的时间）
        double center[3] = {0, 0, 0};   // 旋转中心（相机系，米）
        double yaw = 0.0;               // 该时刻的底盘 yaw（rad，与 EKF 同一相位表）
        double omega = 0.0;             // 角速度（rad/s）
        /// 这条拟合的**参考板绝对编号**（0..3；-1 = 未知）。
        /// 用途：① 诊断"这次拟合算的是哪块板"；② 注入前的相位门控
        /// （拟合 yaw 与 EKF yaw 残差超门限 → 认为参考板号错了 k 号，退回模 90° 吸附）；
        /// ③ 将来启用逐板几何时，按板号判断"yaw"与"yaw+90°"到底哪个才是物理解。
        /// 以前没有这个字段，调用方只能靠"模 90° 吸附"猜，属于治标。
        int ref_armor_id = -1;
        double alpha = 0.0;             // 灯条安装倾角（rad，慢层）
        double R = 0.20;                // 回转半径（m，慢层）
        double h = 0.0;                 // 板心相对旋转中心的高度（m，向上为正；慢层）
        double drift[3] = {0, 0, 0};    // 旋转中心的平移速度（m/s，相机系；快层）
        double reprojection_rmse_px = 0.0;  // 诊断用，不作为唯一有效性判据
        double quality = 0.0;           // 0~1，越大越可信（用于缩放观测协方差）
        int num_bars = 0;               // 参与拟合的灯条数
    };
}

#endif
