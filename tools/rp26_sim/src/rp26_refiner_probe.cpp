// ============================================================================
// 由 RP-26Rune-main/src/core/algorithm/power_rune/src/RuneObservationRefiner.cpp
// 复制而来，**只加日志，不改任何判断逻辑**。
//
// 目的：他们的传统链路（颜色差轮廓 -> 语义分割 -> 描述子可用性 -> 位姿优化）
// 在仿真器画面上的成功率很低（实测 42 帧有检测的帧里只有 6 帧位姿成功），
// 单靠他们的可视化看不出到底卡在哪个描述子上。这里把每个候选的
// "装甲板/灯臂/R标轮廓是否存在 + 描述子数值 + 是否 usable" 打出来，
// 便于对照配置阈值定位误差来源。
// ============================================================================
#include "RuneObservationRefiner.hpp"
#include "json.hpp"
#include "common/PowerRuneDiagnostics.hpp"
#include "img_viz.hpp"
#include <algorithm>
#include <cmath>

RefinedRuneObservation RuneObservationRefiner::refine(const RuneObservation &rune_observation)
{   
    //准备输出结果
    RefinedRuneObservation refined_rune_observation;
    refined_rune_observation.rune_blade_2D.reserve(rune_observation.rune_infos.size());

    //遍历网络结果，进行传统算法的轮廓提取
    for (const auto & rune_info : rune_observation.rune_infos )
    {
        //获取特征轮廓
        std::vector<std::vector<cv::Point>> contours = extract_contours(rune_info,rune_info.color);

        //获取约束过的轮廓
        ConstrainedContours constrained_contours = constrain_contours(rune_info, contours);

        //构造单片符
        SingleRuneBlade2D single_rune_blade_2D = construct_blade(
            rune_observation.is_big_rune,
            rune_info,
            std::move(constrained_contours),
            cv::Rect(0,0,rune_observation.ori_img.size().width,rune_observation.ori_img.size().height)
        );
        
        refined_rune_observation.rune_blade_2D.emplace_back(std::move(single_rune_blade_2D));
    }

    refined_rune_observation.ori_img = rune_observation.ori_img;
    refined_rune_observation.timestamp = rune_observation.timestamp;
    refined_rune_observation.tf_tree = rune_observation.tf_tree;
    refined_rune_observation.is_big_rune = rune_observation.is_big_rune;
    return refined_rune_observation;

}

std::vector<std::vector<cv::Point>> RuneObservationRefiner::extract_contours(const RuneInfo &rune_info, RuneInfo::Color color)
{   
    const cv::Mat &view = rune_info.view;
    auto view_x_offset = rune_info.view_rect.x;
    auto view_y_offset = rune_info.view_rect.y;

    //通道分离
    std::vector<cv::Mat> bgr;
    cv::split(view, bgr);

    //获取差值图，滤波，二值化
    cv::Mat minus_mat = color == RuneInfo::RED ? bgr[2] - bgr[0] : bgr[0] - bgr[2];
    cv::GaussianBlur(minus_mat, minus_mat, cv::Size(5, 5), 0);
    cv::Mat minus_mat_mask;
    double bin_threshold = color == RuneInfo::RED
                               ? (double)J_POWER_RUNE.config_["feature_extract"]["red_minus_blue2bin_threshold"]
                               : (double)J_POWER_RUNE.config_["feature_extract"]["blue_minus_red2bin_threshold"];
    cv::threshold(minus_mat, minus_mat_mask, bin_threshold, 255, cv::THRESH_BINARY);

    // 提取轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(minus_mat_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if ((int)J_POWER_RUNE.config_["debug"]["POWER_RUNE_DEBUG"] &&
        (int)J_POWER_RUNE.config_["debug"]["POWER_RUNE_DEBUG_DETECT"] &&
        (int)J_POWER_RUNE.config_["debug"]["POWER_RUNE_DEBUG_DETECT_SHOW_BIN_IMG"])
    {
        ImgViz::enqueue_image_zero_copy("PowerRune/BinaryMask", minus_mat_mask);
    }
    // 还原到原图尺度
    for (auto &contour : contours)
    {
        for (auto &pt : contour)
        {
            pt.x += view_x_offset;
            pt.y += view_y_offset;
        }
    }

    // 返回轮廓
    return contours;
}

ConstrainedContours RuneObservationRefiner::constrain_contours(const RuneInfo &rune_info, const std::vector<std::vector<cv::Point>> &contours)
{

    //没有轮廓直接返回空值
    if (contours.empty())
    {
        return ConstrainedContours();
    }

    //网络认为的装甲板模块的中心点
    cv::Point2f nn_armor_module_center = (rune_info.top + rune_info.left + rune_info.bottom + rune_info.right) / 4.f;
    
    //网络认为的装甲板模块对应的椭圆面积(a * b * pi)
    double nn_ellipse_area = 0.25f * cv::norm(rune_info.top - rune_info.bottom) * cv::norm(rune_info.left - rune_info.right) * CV_PI;

    // 装甲板模块的过滤条件
    auto armor_module_filter = [&](const std::vector<cv::Point> &contour) -> bool
    {
        double contour_area = cv::contourArea(contour);
        double rel_err = std::fabs(contour_area - nn_ellipse_area) / nn_ellipse_area;

        bool is_inside = cv::pointPolygonTest(contour, nn_armor_module_center, false) > 0;
        bool is_acceptable_area = rel_err <= (double)J_POWER_RUNE.config_["feature_constrain"]["armor_module_area_relative_error_threshold"];

        return is_inside && is_acceptable_area;
    };

    //灯臂的过滤条件
    auto light_arm_filter = [&](const std::vector<cv::Point> &contour)-> bool
    {
    
        //判断nn_armor_module_center是否在contour外
        if(!(cv::pointPolygonTest(contour, nn_armor_module_center, false) < 0))
        {
            //不在外部说明不是灯臂
            return false;
        }

        //判断rune_info.point_R是否在contour外
        if(!(cv::pointPolygonTest(contour, rune_info.point_R, false) < 0))
        {   
            //不在外部说明不是灯臂
            return false;
        }
        
        //判断这R标圆心和装甲板模块的连线是否经过contour
        if(!is_line_pass_through_contour(nn_armor_module_center, rune_info.point_R, contour, (int)J_POWER_RUNE.config_["feature_constrain"]["light_arm_line_samples"]))
        {   
            //如果连线没有穿过说明轮廓可能太小了或者不在轴线上，说明不是灯臂
            return false;
        }

        //判断装甲模块的左右端点连线是否经过contour
        if(is_line_pass_through_contour(rune_info.left,rune_info.right,contour,(int)J_POWER_RUNE.config_["feature_constrain"]["light_arm_line_samples"]))
        {
            //如果连线穿过了说明是一个极端的装甲板轮廓(月牙形状的)，不是灯臂
            return false;
        }

        //返回标志位
        return true;

    };

    //R标的过滤条件
    auto center_R_filter = [&](const std::vector<cv::Point> &contour)-> bool
    {   
        //如果网络认为的R标志不在轮廓内说明不是
        if (!(cv::pointPolygonTest(contour, rune_info.point_R, false) > 0))
        {
            return false;
        }

        //网络识别的所有其他的点以及装甲板模块的中心点都不能在轮廓内
        bool is_other_point_outside;
        is_other_point_outside = 
        (cv::pointPolygonTest(contour, rune_info.top, false) < 0)
        && (cv::pointPolygonTest(contour, rune_info.left, false) < 0)
        && (cv::pointPolygonTest(contour, rune_info.bottom, false) < 0)
        && (cv::pointPolygonTest(contour, rune_info.right, false) < 0)
        && (cv::pointPolygonTest(contour, nn_armor_module_center, false) < 0);


        return is_other_point_outside;
    };

    /*
    上述三个 filter 同时承担“分类 + 过滤”的职责。
    通过对关键语义点（装甲中心、R 点）的包含/排斥约束，
    人为构造了三类轮廓之间的互斥关系：
    
    1. armor_module 必须包含 nn_armor_module_center，
       而 light_arm 明确要求该点在轮廓外，
       因此同一轮廓不会同时被判为装甲板和灯臂。
    
    2. center_R 必须包含 rune_info.point_R，
       而 light_arm 要求该点在轮廓外，
       因此同一轮廓不会同时被判为 R 标和灯臂。
    
    3. armor_module 必须包含 nn_armor_module_center，
       而 center_R 明确要求该点在轮廓外，
       因此同一轮廓不会同时被判为装甲板和 R 标。
    */

    //利用上述三个滤波器过滤
    std::vector<std::vector<cv::Point>> armor_module_contour_candidate;
    std::vector<std::vector<cv::Point>> light_arm_contour_candidate;
    std::vector<std::vector<cv::Point>> center_R_contour_candidate;

    //TODO：这里可以用或运算加速，但是为了方便debug所以没有这么写
    for (const auto &contour : contours)
    {
        if (contour.size() <= 5)
        {
            //必须保证至少有5个点
            continue;
        }
        

        bool is_armor_module = armor_module_filter(contour);
        bool is_light_arm = light_arm_filter(contour);
        bool is_center_R = center_R_filter(contour);

        if ((int)is_armor_module + (int)is_light_arm + (int)is_center_R > 1)
        {
            LOG(ERROR) << "[constrain_contours]约束关系错误";
            return ConstrainedContours();
        }

        if (is_armor_module)
        {
            armor_module_contour_candidate.push_back(contour);
        }
        else if (is_light_arm)
        {
            light_arm_contour_candidate.push_back(contour);

        }
        else if (is_center_R)
        {
            center_R_contour_candidate.push_back(contour);
        }
    }

    // 运行到此处已经分类并且过滤完毕。接下来需要保证唯一性
    ConstrainedContours constrained_contours;

    // 对于装甲板模块，应该优先选择和网络相对误差最小的
    if (!armor_module_contour_candidate.empty())
    {
        // 只有一个不做筛选
        if (armor_module_contour_candidate.size() == 1)
        {
            constrained_contours.armor_module_opt = std::move(armor_module_contour_candidate[0]);
        }
        else
        {
            // 有多个时选择相对误差最小的
            double min_rel_err = std::numeric_limits<double>::max();
            size_t best_idx = 0;

            for (size_t i = 0; i < armor_module_contour_candidate.size(); ++i)
            {
                double contour_area = cv::contourArea(armor_module_contour_candidate[i]);
                double rel_err = std::fabs(contour_area - nn_ellipse_area) / nn_ellipse_area;

                if (rel_err < min_rel_err)
                {
                    min_rel_err = rel_err;
                    best_idx = i;
                }
            }

            constrained_contours.armor_module_opt = std::move(armor_module_contour_candidate[best_idx]);
        }
    }

    // 对于灯臂，应该优先选择面积最大的
    if (!light_arm_contour_candidate.empty())
    {
        // 只有一个不做筛选
        if (light_arm_contour_candidate.size() == 1)
        {
            constrained_contours.light_arm_opt = std::move(light_arm_contour_candidate[0]);
        }
        else
        {
            // 有多个的时候选择面积最大的
            double max_area = 0.0;
            size_t best_idx = 0;

            for (size_t i = 0; i < light_arm_contour_candidate.size(); ++i)
            {
                double contour_area = cv::contourArea(light_arm_contour_candidate[i]);

                if (contour_area > max_area)
                {
                    max_area = contour_area;
                    best_idx = i;
                }
            }

            constrained_contours.light_arm_opt = std::move(light_arm_contour_candidate[best_idx]);
        }
    }

    // 对于R标，应该优先选择面积最小的
    if (!center_R_contour_candidate.empty())
    {
        // 只有一个不做筛选
        if (center_R_contour_candidate.size() == 1)
        {
            constrained_contours.center_R_opt = std::move(center_R_contour_candidate[0]);
        }
        else
        {
            // 有多个的时候选择面积最小的
            double min_area = std::numeric_limits<double>::max();
            size_t best_idx = 0;

            for (size_t i = 0; i < center_R_contour_candidate.size(); ++i)
            {
                double contour_area = cv::contourArea(center_R_contour_candidate[i]);

                if (contour_area < min_area)
                {
                    min_area = contour_area;
                    best_idx = i;
                }
            }

            constrained_contours.center_R_opt = std::move(center_R_contour_candidate[best_idx]);
        }
    }

    //返回结果
    return constrained_contours; 
}

namespace
{
// 与 Projector::is_valid_for_projection（PowerRunePlane.cpp:1795）等价，仅用于日志。
bool is_valid_for_projection_probe(const SingleRuneBlade2D &blade)
{
    if (blade.rune_state == RuneState::BigInactive || blade.rune_state == RuneState::SmallInactive ||
        blade.rune_state == RuneState::SmallActivated)
        return blade.is_armor_module_usable && blade.is_light_arm_usable &&
               blade.is_center_R_usable;
    return blade.is_center_R_usable && blade.is_light_arm_usable;
}
} // namespace

SingleRuneBlade2D RuneObservationRefiner::construct_blade(bool is_big_rune, const RuneInfo &rune_info, ConstrainedContours &&constrained_contours, const cv::Rect &ori_img_rect)
{
    SingleRuneBlade2D single_rune_blade_2D;
    single_rune_blade_2D.rune_state = is_big_rune
                                          ? (rune_info.class_id == 0 ? RuneState::BigInactive : RuneState::BigActivated)
                                          : (rune_info.class_id == 0 ? RuneState::SmallInactive : RuneState::SmallActivated);
    single_rune_blade_2D.constrained_contours = std::move(constrained_contours);
    single_rune_blade_2D.rune_info = rune_info;

    //对轮廓进行是否可以进行重投影作判断
    auto &contours = single_rune_blade_2D.constrained_contours;
    // 装甲板模块
    double probe_armor_solidity = -1.0;
    bool probe_armor_border = true;
    double probe_arm_solidity = -1.0;
    float probe_arm_aspect = -1.0F;
    if (contours.armor_module_opt.has_value())
    {
        EllipseDescriptor ellipse_descriptor(contours.armor_module_opt.value(), ori_img_rect);
        single_rune_blade_2D.is_armor_module_usable = ellipse_descriptor.is_usable();
        probe_armor_solidity = ellipse_descriptor.get_solidity();
        probe_armor_border = ellipse_descriptor.is_near_image_border();
    }

    // 灯臂
    if (contours.light_arm_opt.has_value())
    {
        if (rune_info.class_id == 0)
        {
            RectangularDescriptor rectangular_descriptor(contours.light_arm_opt.value(), ori_img_rect);
            single_rune_blade_2D.is_light_arm_usable = rectangular_descriptor.is_usable();
            probe_arm_solidity = rectangular_descriptor.get_solidity();
            probe_arm_aspect = rectangular_descriptor.get_aspect_ratio();
        }
        else
        {
            PeakProfileDescriptor peak_profile_descriptor(
                contours.light_arm_opt.value(),
                ori_img_rect,
                is_big_rune);
            single_rune_blade_2D.is_light_arm_usable = peak_profile_descriptor.is_usable();
        }
    }

    // R标
    single_rune_blade_2D.is_center_R_usable = contours.center_R_opt.has_value();

    LOG(INFO) << "[REFINE_PROBE] class=" << rune_info.class_id
              << " armor=" << (contours.armor_module_opt.has_value() ? 1 : 0)
              << "(usable=" << single_rune_blade_2D.is_armor_module_usable
              << ",solidity=" << probe_armor_solidity << ",border=" << probe_armor_border << ")"
              << " lightarm=" << (contours.light_arm_opt.has_value() ? 1 : 0)
              << "(usable=" << single_rune_blade_2D.is_light_arm_usable
              << ",solidity=" << probe_arm_solidity << ",aspect=" << probe_arm_aspect << ")"
              << " centerR=" << (contours.center_R_opt.has_value() ? 1 : 0)
              << " -> projectable=" << is_valid_for_projection_probe(single_rune_blade_2D);

    return single_rune_blade_2D;
}

bool RuneObservationRefiner::is_line_pass_through_contour(const cv::Point2f &A, const cv::Point2f &B, const std::vector<cv::Point> &contour, int samples)
{

    for (int i = 0; i <= samples; ++i)
    {   
        //更具采样次数计算采样点的位置
        float t = static_cast<float>(i) / samples;
        cv::Point2f P = A + t * (B - A);

        //判断是否在内部
        if (cv::pointPolygonTest(contour, P, false) > 0)
        {
            return true;
        }
    }

    return false;
}

void RuneObservationRefiner::rebuild_rectangle(std::vector<cv::Point> &contour, cv::Rect ori_img_rect)
{
    struct ProjectedPoint
    {
        cv::Point2f img_pt;
        float x_local;
        float y_local;
    };

    // 至少需要足够多的边界点，四条边才能稳定分组和拟合。
    if (contour.size() < 8 || ori_img_rect.width <= 0 || ori_img_rect.height <= 0)
    {
        return;
    }

    // 轮廓点整理成 PCA 输入矩阵，每一行对应一个像素点的图像坐标。
    cv::Mat data_pts(static_cast<int>(contour.size()), 2, CV_64F);
    for (int i = 0; i < static_cast<int>(contour.size()); ++i)
    {
        data_pts.at<double>(i, 0) = static_cast<double>(contour[i].x);
        data_pts.at<double>(i, 1) = static_cast<double>(contour[i].y);
    }

    // PCA 在这里不是为了直接构造“旋转矩形”，而是为了给当前轮廓建立一个
    // 局部坐标系。这样可以把“哪一些点更靠近四条边”这个问题，转成在主轴/
    // 副轴方向上的一维投影问题。
    const cv::PCA pca(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
    const cv::Point2f center(
        static_cast<float>(pca.mean.at<double>(0, 0)),
        static_cast<float>(pca.mean.at<double>(0, 1)));

    cv::Point2f axis_x(
        static_cast<float>(pca.eigenvectors.at<double>(0, 0)),
        static_cast<float>(pca.eigenvectors.at<double>(0, 1)));
    const float axis_x_norm = std::sqrt(axis_x.dot(axis_x));
    if (axis_x_norm <= 1e-6f)
    {
        return;
    }
    axis_x *= (1.0f / axis_x_norm);
    const cv::Point2f axis_y(-axis_x.y, axis_x.x);

    // 把轮廓点投影到 PCA 局部坐标系中：
    // - x_local 表示点沿主方向的位置；
    // - y_local 表示点沿副方向的位置。
    // 后面会在这两个坐标上取分位数，并据此选出靠近四条边的点带。
    std::vector<ProjectedPoint> projected_pts;
    std::vector<float> proj_x;
    std::vector<float> proj_y;
    projected_pts.reserve(contour.size());
    proj_x.reserve(contour.size());
    proj_y.reserve(contour.size());
    for (const auto &point : contour)
    {
        const cv::Point2f img_pt(static_cast<float>(point.x), static_cast<float>(point.y));
        const cv::Point2f vec = img_pt - center;
        const float x_local = vec.dot(axis_x);
        const float y_local = vec.dot(axis_y);

        projected_pts.push_back({img_pt, x_local, y_local});
        proj_x.emplace_back(x_local);
        proj_y.emplace_back(y_local);
    }

    // 计算一维投影分位数。nth_element 只关心局部顺序，足够拿到需要的分位点。
    auto quantile = [](std::vector<float> values, float q) -> float
    {
        if (values.empty())
        {
            return 0.0f;
        }

        q = std::clamp(q, 0.0f, 1.0f);
        const float index = q * static_cast<float>(values.size() - 1);
        const size_t lower = static_cast<size_t>(std::floor(index));
        const size_t upper = static_cast<size_t>(std::ceil(index));

        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lower), values.end());
        const float lower_value = values[lower];
        if (upper == lower)
        {
            return lower_value;
        }

        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(upper), values.end());
        const float upper_value = values[upper];
        return lower_value + (index - static_cast<float>(lower)) * (upper_value - lower_value);
    };

    // 这里仍然用分位数来定义“主体区域”的左右/上下范围，而不是直接拿
    // min/max。这样能过滤掉局部毛刺、轻微凹凸和离群像素，避免拟合出来
    // 的边线被少量异常点带偏。
    constexpr float kLowerQuantile = 0.00f;
    constexpr float kUpperQuantile = 1.00f;
    const float x_lo = quantile(proj_x, kLowerQuantile);
    const float x_hi = quantile(proj_x, kUpperQuantile);
    const float y_lo = quantile(proj_y, kLowerQuantile);
    const float y_hi = quantile(proj_y, kUpperQuantile);

    const float width_local = x_hi - x_lo;
    const float height_local = y_hi - y_lo;
    if (width_local <= 1.0f || height_local <= 1.0f)
    {
        return;
    }

    // 只取靠近四条边的一条“窄带”来拟合直线，而不是拿整片轮廓全部去拟合。
    // 这样能尽量减少中间区域和转角区域对边线估计的影响。
    const float x_band = std::max(1.5f, width_local * 0.18f);
    const float y_band = std::max(1.5f, height_local * 0.18f);

    std::vector<cv::Point2f> left_pts;
    std::vector<cv::Point2f> right_pts;
    std::vector<cv::Point2f> top_pts;
    std::vector<cv::Point2f> bottom_pts;
    left_pts.reserve(contour.size() / 2);
    right_pts.reserve(contour.size() / 2);
    top_pts.reserve(contour.size() / 2);
    bottom_pts.reserve(contour.size() / 2);

    for (const auto &pt : projected_pts)
    {
        if (pt.x_local <= x_lo + x_band)
        {
            left_pts.push_back(pt.img_pt);
        }
        if (pt.x_local >= x_hi - x_band)
        {
            right_pts.push_back(pt.img_pt);
        }
        if (pt.y_local <= y_lo + y_band)
        {
            top_pts.push_back(pt.img_pt);
        }
        if (pt.y_local >= y_hi - y_band)
        {
            bottom_pts.push_back(pt.img_pt);
        }
    }

    if (left_pts.size() < 2 || right_pts.size() < 2 || top_pts.size() < 2 || bottom_pts.size() < 2)
    {
        return;
    }

    auto fit_line = [](const std::vector<cv::Point2f> &pts) -> cv::Vec4f
    {
        cv::Vec4f line;
        cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
        return line;
    };

    auto intersect_lines = [](const cv::Vec4f &line1, const cv::Vec4f &line2, cv::Point2f &cross_pt) -> bool
    {
        const cv::Point2f d1(line1[0], line1[1]);
        const cv::Point2f p1(line1[2], line1[3]);
        const cv::Point2f d2(line2[0], line2[1]);
        const cv::Point2f p2(line2[2], line2[3]);

        const float det = d1.x * d2.y - d1.y * d2.x;
        if (std::abs(det) <= 1e-4f)
        {
            return false;
        }

        const cv::Point2f diff = p2 - p1;
        const float t = (diff.x * d2.y - diff.y * d2.x) / det;
        cross_pt = p1 + t * d1;
        return true;
    };

    // 四条边在图像里不再假设互相垂直，而是分别独立拟合成 4 条直线。
    // 这样重建出来的是一个透视下的凸四边形，比旋转矩形更符合真实成像。
    const cv::Vec4f left_line = fit_line(left_pts);
    const cv::Vec4f right_line = fit_line(right_pts);
    const cv::Vec4f top_line = fit_line(top_pts);
    const cv::Vec4f bottom_line = fit_line(bottom_pts);

    cv::Point2f top_left;
    cv::Point2f top_right;
    cv::Point2f bottom_right;
    cv::Point2f bottom_left;
    if (!intersect_lines(left_line, top_line, top_left)
        || !intersect_lines(right_line, top_line, top_right)
        || !intersect_lines(right_line, bottom_line, bottom_right)
        || !intersect_lines(left_line, bottom_line, bottom_left))
    {
        return;
    }

    auto clamp_to_image = [&](const cv::Point2f &pt) -> cv::Point2f
    {
        return cv::Point2f(
            static_cast<float>(std::clamp(static_cast<int>(std::lround(pt.x)),
                                          ori_img_rect.x,
                                          ori_img_rect.x + ori_img_rect.width - 1)),
            static_cast<float>(std::clamp(static_cast<int>(std::lround(pt.y)),
                                          ori_img_rect.y,
                                          ori_img_rect.y + ori_img_rect.height - 1)));
    };

    const std::vector<cv::Point2f> corners = {
        clamp_to_image(top_left),
        clamp_to_image(top_right),
        clamp_to_image(bottom_right),
        clamp_to_image(bottom_left)};

    // 交点如果几乎退化成一条线，说明本帧不适合做四边形重建。
    if (std::abs(cv::contourArea(corners)) < 4.0)
    {
        return;
    }

    // 后续模块需要的仍然是一圈边界点，而不是 4 个角点，所以最后沿四条边
    // 做稠密重采样，把四边形重新变回 contour 形式。
    auto append_edge = [](std::vector<cv::Point> &dense_contour, const cv::Point2f &start, const cv::Point2f &end)
    {
        const float dx = end.x - start.x;
        const float dy = end.y - start.y;
        const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)))));

        for (int i = 0; i <= steps; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(steps);
            const cv::Point sample(
                static_cast<int>(std::lround(start.x + t * dx)),
                static_cast<int>(std::lround(start.y + t * dy)));
            if (dense_contour.empty() || dense_contour.back() != sample)
            {
                dense_contour.emplace_back(sample);
            }
        }
    };

    std::vector<cv::Point> dense_contour;
    dense_contour.reserve(static_cast<size_t>(2.0f * (width_local + height_local) + 8.0f));
    for (size_t i = 0; i < corners.size(); ++i)
    {
        append_edge(dense_contour, corners[i], corners[(i + 1) % corners.size()]);
    }

    if (dense_contour.size() >= 4)
    {
        contour = std::move(dense_contour);
    }
}

ContourDescriptor::ContourDescriptor(std::vector<cv::Point> contour, cv::Rect ori_img_rect)
    : m_contour(std::move(contour)),
      m_ori_img_rect(std::move(ori_img_rect))
{
}

bool ContourDescriptor::is_contour_near_image_border()
{   
    int img_width = m_ori_img_rect.width;
    int img_height = m_ori_img_rect.height;

    for (const auto &point : m_contour)
    {   
        //如果离边界小于2个像素点就可以认为轮廓在边缘上
        int distance = std::min({point.x, point.y, img_width - 1 - point.x, img_height - 1 - point.y});

        if (distance <= 2)
        {
            return true;
        }
          
    }
    
    return false;
}

EllipseDescriptor::EllipseDescriptor(std::vector<cv::Point> contour, cv::Rect ori_img_rect)
    : ContourDescriptor(std::move(contour),std::move(ori_img_rect))
{
    //实心度，判断是否因为遮挡或者破碎产生了明显的内凹；
    double contour_area = cv::contourArea(m_contour);
    std::vector<cv::Point> hull;
    cv::convexHull(m_contour, hull);
    double hull_area = cv::contourArea(hull);
    m_solidity = contour_area / hull_area;
    m_is_near_image_border =  is_contour_near_image_border();

}

bool EllipseDescriptor::is_usable() const
{   

    if (m_solidity <= (double)J_POWER_RUNE.config_["contour_descriptor"]["solidity_threshold_ellipse"])
    {
        //说明凹陷比较严重
        return false;
    }

    //感觉这个比较耗时，不一定有必要用
    if (m_is_near_image_border)
    {
        //说明图像靠近边缘
        return false;
    }
    
    return true;
    
}

double EllipseDescriptor::get_solidity() const
{
    return m_solidity;
}

bool EllipseDescriptor::is_near_image_border() const
{
    return m_is_near_image_border;
}

RectangularDescriptor::RectangularDescriptor(std::vector<cv::Point> contour, cv::Rect ori_img_rect)
    : ContourDescriptor(std::move(contour),std::move(ori_img_rect))
{
    //实心度，判断是否因为遮挡或者破碎产生了明显的内凹；
    double contour_area = cv::contourArea(m_contour);
    std::vector<cv::Point> hull;
    cv::convexHull(m_contour, hull);
    double hull_area = cv::contourArea(hull);
    m_solidity = contour_area / hull_area;

    //长宽比
    cv::RotatedRect min_area_rect = minAreaRect(m_contour);
    m_aspect_ratio = std::max(min_area_rect.size.width, min_area_rect.size.height) / std::min(min_area_rect.size.width, min_area_rect.size.height);
}

bool RectangularDescriptor::is_usable() const
{
    if (m_solidity <= (double)J_POWER_RUNE.config_["contour_descriptor"]["solidity_threshold_rectangular"])
    {
        //说明凹陷比较严重
        return false;
    }

    float aspect_ratio_relative_error = fabs((float)J_POWER_RUNE.config_["contour_descriptor"]["expect_aspect_ratio"] - m_aspect_ratio) / (float)J_POWER_RUNE.config_["contour_descriptor"]["expect_aspect_ratio"];
    if (aspect_ratio_relative_error >= (float)J_POWER_RUNE.config_["contour_descriptor"]["aspect_ratio_relative_error_threshold"])
    {
        //长宽比与期望差距过大，说明有非常大的动态模糊或者被遮挡
        return false;

    }

    //不必判断是否在边缘，因为必然是装甲板模块不在边缘
    return true;
}

double RectangularDescriptor::get_solidity() const
{
    return m_solidity;
}

float RectangularDescriptor::get_aspect_ratio() const
{
    return m_aspect_ratio;
}

PeakProfileDescriptor::PeakProfileDescriptor(std::vector<cv::Point> contour, cv::Rect ori_img_rect, bool is_big_rune)
    : ContourDescriptor(std::move(contour), std::move(ori_img_rect)),
      m_is_big_rune(is_big_rune)
{   
    //计算外接多边形轮廓
    double arc_len = cv::arcLength(m_contour, true);
    double epsilon = (double)J_POWER_RUNE.config_["contour_descriptor"]["approx_error_tolerance"] * arc_len;
    cv::approxPolyDP(m_contour, m_approx_contour, epsilon, true);
    m_approx_contour_size = m_approx_contour.size();

    
    if (is_big_rune)
    {
        //对于大符来说，拐点的数目应该在8-16之间
        if (m_approx_contour_size >= 8 && m_approx_contour_size <= 16)
        {
            m_is_approx_contour_size_valid = true;
        }
        else
        {
            m_is_approx_contour_size_valid = false;
        }

        m_is_near_image_border = is_contour_near_image_border();
    }
    else
    {
        //对于小符来说，拐点的数目应该在12-16之间
        if (m_approx_contour_size >= 12 && m_approx_contour_size <= 16)
        {
            m_is_approx_contour_size_valid = true;
        }
        else
        {
            m_is_approx_contour_size_valid = false;
        }

        //小符不关心是否在边缘
        m_is_near_image_border = false;
    }
    


}

bool PeakProfileDescriptor::is_usable() const
{
    if (!m_is_approx_contour_size_valid)
    {   
        //拐点不满足条件
        return false;
    }
    
    if (m_is_near_image_border)
    {
        //太靠近边缘
        return false;
    }
    
    return true;
    
}

const std::vector<cv::Point> PeakProfileDescriptor::get_m_approx_contour() const
{
    return m_approx_contour;
}

const int PeakProfileDescriptor::get_approx_contour_size() const
{
    return m_approx_contour_size;
}

bool PeakProfileDescriptor::is_big_rune() const
{
    return m_is_big_rune;
}

bool PeakProfileDescriptor::is_approx_contour_size_valid() const
{
    return m_is_approx_contour_size_valid;
}

bool PeakProfileDescriptor::is_near_image_border() const
{
    return m_is_near_image_border;
}
