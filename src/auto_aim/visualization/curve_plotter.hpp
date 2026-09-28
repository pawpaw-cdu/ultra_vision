#ifndef ULTRA_VISION_AUTO_AIM_VISUALIZATION_CURVE_PLOTTER_HPP
#define ULTRA_VISION_AUTO_AIM_VISUALIZATION_CURVE_PLOTTER_HPP

// 实时数学曲线（对齐 sp_vision / jlu_vision 的调试习惯：**看曲线**，不是只看图）。
//
// 为什么需要：图像上的框只能告诉你"这一帧对上了"，而"装甲板位姿与瞄准方向是否
// 一直对得上"是个**时间序列**问题 —— 滞后、超前、跳变、偏置都只在曲线上看得见。
//
// 布局：曲线按 panel 分组，**同一 panel 里的曲线共用一条纵轴**（这样两条线的
// 高低差才有意义），图例在 panel 内**每行一条**（以前两条曲线把文字画在同一个
// 坐标上，字叠在一起看不清）。`renderBeside()` 直接把曲线画到图像右边，
// 上层只要 imshow 一次 —— 原来 Detection / Curves 两个窗口现在是一个。
//
// 曲线可以同时落 CSV（ULTRA_VISION_CURVE_CSV），方便离线重画或和真值对齐。

#include <algorithm>
#include <cstdio>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace auto_aim::visualization
{
    class CurvePlotter
    {
    public:
        struct Curve
        {
            std::string name;
            cv::Scalar color;
            std::deque<double> values;
            bool auto_scale = true;
            double minimum = 0.0;
            double maximum = 1.0;
            int panel = 0;
        };

        explicit CurvePlotter(std::size_t history = 300) : history_(history) {}

        /// @brief 注册一条曲线。
        /// @param span 手动纵轴半幅（0 表示按数据自动）；两条同量纲曲线用同一个 span
        ///             就会落在同一个 panel 里。
        void addCurve(const std::string& name, const cv::Scalar& color, double span = 0.0,
                      int panel = -1)
        {
            Curve curve;
            curve.name = name;
            curve.color = color;
            curve.auto_scale = span <= 0.0;
            curve.maximum = span;
            curve.minimum = -span;
            curve.panel = panel >= 0
                ? panel
                : static_cast<int>(panelsUsed_++ / kCurvesPerPanel);
            curves_.push_back(std::move(curve));
        }

        void push(double value)
        {
            if (active_ < curves_.size()) curves_[active_].values.push_back(value);
        }

        /// @brief 一条曲线一组值：`pushAll({a, b, c})` 对应按注册顺序的曲线。
        void pushAll(const std::vector<double>& values)
        {
            for (std::size_t i = 0; i < values.size() && i < curves_.size(); ++i) {
                curves_[i].values.push_back(values[i]);
                while (curves_[i].values.size() > history_) curves_[i].values.pop_front();
            }
            if (csv_.is_open()) {
                for (std::size_t i = 0; i < values.size(); ++i) {
                    csv_ << (i == 0 ? "" : ",") << values[i];
                }
                csv_ << '\n';
            }
        }

        bool openCsv(const std::string& path)
        {
            csv_.open(path, std::ios::out | std::ios::trunc);
            if (!csv_.is_open()) return false;
            for (std::size_t i = 0; i < curves_.size(); ++i) {
                csv_ << (i == 0 ? "" : ",") << curves_[i].name;
            }
            csv_ << '\n';
            return true;
        }

        int panelCount() const
        {
            int count = 1;
            for (const Curve& curve : curves_) count = std::max(count, curve.panel + 1);
            return count;
        }

        /// @brief 画成一幅图（每个 panel 一行，panel 内的曲线共用纵轴）。
        cv::Mat render(int width = 720, int row_height = 110) const
        {
            if (curves_.empty()) return {};
            const int panels = panelCount();
            cv::Mat canvas(panels * row_height, width, CV_8UC3, background_);
            for (int panel = 0; panel < panels; ++panel) {
                std::vector<const Curve*> members;
                for (const Curve& curve : curves_) {
                    if (curve.panel == panel) members.push_back(&curve);
                }
                cv::Mat view = canvas(cv::Rect(0, panel * row_height, width, row_height));
                drawPanel(view, members, cv::Size(width, row_height));
                if (panel > 0) {
                    cv::line(canvas, cv::Point(0, panel * row_height),
                             cv::Point(width, panel * row_height), cv::Scalar(60, 60, 64), 1);
                }
            }
            return canvas;
        }

        /// @brief 曲线拼到图像**下方**，返回一张合成图（上层 imshow 一次即可）。
        ///
        /// 竖直组合：图像在上、曲线在下，两者同宽（都是 image.cols），
        /// 每条曲线在整幅宽度上展开，比并排时更容易看相位差。
        /// `max_total_height > 0` 时自动压每格高度，避免窗口比屏幕还高
        /// （默认上限 900 px：笔记本 1440x900 也能整屏放下）。
        cv::Mat renderStacked(const cv::Mat& image, int max_total_height = 900,
                              int row_height = 0) const
        {
            const int panels = std::max(panelCount(), 1);
            if (image.empty()) return render(720, row_height > 0 ? row_height : 110);
            if (row_height <= 0) {
                row_height = 110;
                if (max_total_height > 0 && image.rows + panels * row_height > max_total_height) {
                    row_height = (max_total_height - image.rows) / panels;
                }
                row_height = std::max(row_height, 70);
            }
            const int width = image.cols;
            cv::Mat canvas = render(width, row_height);
            cv::Mat merged;
            cv::vconcat(image, canvas, merged);
            cv::line(merged, cv::Point(0, image.rows), cv::Point(width - 1, image.rows),
                     cv::Scalar(90, 90, 90), 2);
            return merged;
        }

        /// @brief 曲线画到图像右侧（另一种排法，保留给宽屏 / 截图用）。
        cv::Mat renderBeside(const cv::Mat& image) const
        {
            const int panels = panelCount();
            if (image.empty()) return render(720, 110);
            const int width = image.cols;
            const int row_height = std::max(60, image.rows / std::max(panels, 1));
            cv::Mat canvas = render(width, row_height);
            cv::Mat fitted(image.rows, width, CV_8UC3, background_);
            const int usable = std::min(image.rows, canvas.rows);
            canvas(cv::Rect(0, 0, width, usable)).copyTo(fitted(cv::Rect(0, 0, width, usable)));
            cv::Mat merged;
            cv::hconcat(image, fitted, merged);
            // 图像区 / 曲线区之间的分隔线，避免两边贴在一起看串行。
            cv::line(merged, cv::Point(image.cols, 0), cv::Point(image.cols, merged.rows - 1),
                     cv::Scalar(90, 90, 90), 2);
            return merged;
        }

    private:
        static constexpr int kCurvesPerPanel = 2;
        static inline const cv::Scalar background_{28, 28, 32};

        static void drawPanel(cv::Mat panel, const std::vector<const Curve*>& members,
                              const cv::Size& size)
        {
            const int margin = 6;
            const int legend_lines = static_cast<int>(members.size());
            const int legend_height = 16 * std::max(legend_lines, 1) + 4;
            const int plot_top = legend_height;
            const int plot_height = size.height - plot_top - margin - 4;
            const int plot_width = size.width - 2 * margin - 34;
            if (plot_width < 20 || plot_height < 10 || members.empty()) return;

            // 纵轴范围：同 panel 的曲线共用一个（这是"两条线放在一起比较"的前提）。
            double minimum = 0.0;
            double maximum = 0.0;
            bool have_range = false;
            for (const Curve* curve : members) {
                double lo = curve->minimum;
                double hi = curve->maximum;
                if (curve->auto_scale) {
                    if (curve->values.empty()) continue;
                    lo = *std::min_element(curve->values.begin(), curve->values.end());
                    hi = *std::max_element(curve->values.begin(), curve->values.end());
                }
                minimum = have_range ? std::min(minimum, lo) : lo;
                maximum = have_range ? std::max(maximum, hi) : hi;
                have_range = true;
            }
            if (!have_range) { minimum = -1.0; maximum = 1.0; }
            if (maximum - minimum < 1e-6) {
                const double mid = 0.5 * (minimum + maximum);
                minimum = mid - 0.5;
                maximum = mid + 0.5;
            }
            const double pad = 0.05 * (maximum - minimum);
            minimum -= pad;
            maximum += pad;
            const double span = maximum - minimum;

            const int left = margin + 34;
            const int bottom = plot_top + plot_height;
            cv::line(panel, cv::Point(left, plot_top), cv::Point(left, bottom),
                     cv::Scalar(70, 70, 70), 1);
            cv::line(panel, cv::Point(left, bottom), cv::Point(left + plot_width, bottom),
                     cv::Scalar(70, 70, 70), 1);
            // 3 条横向网格 + 零线（范围跨 0 时才画）。
            for (int i = 1; i <= 3; ++i) {
                const int y = plot_top + plot_height * i / 4;
                cv::line(panel, cv::Point(left, y), cv::Point(left + plot_width, y),
                         cv::Scalar(48, 48, 52), 1);
            }
            if (minimum < 0.0 && maximum > 0.0) {
                const int zero = bottom - static_cast<int>(
                    plot_height * (0.0 - minimum) / span);
                cv::line(panel, cv::Point(left, zero), cv::Point(left + plot_width, zero),
                         cv::Scalar(110, 110, 110), 1);
            }
            char text[160];
            std::snprintf(text, sizeof(text), "%.2f", maximum);
            cv::putText(panel, text, cv::Point(2, plot_top + 8), cv::FONT_HERSHEY_SIMPLEX, 0.32,
                        cv::Scalar(140, 140, 140), 1);
            std::snprintf(text, sizeof(text), "%.2f", minimum);
            cv::putText(panel, text, cv::Point(2, bottom), cv::FONT_HERSHEY_SIMPLEX, 0.32,
                        cv::Scalar(140, 140, 140), 1);

            int legend_y = 14;
            for (const Curve* curve : members) {
                const int count = static_cast<int>(curve->values.size());
                if (count >= 2) {
                    for (int i = 1; i < count; ++i) {
                        const double x0 = left + plot_width * static_cast<double>(i - 1) /
                            (count - 1);
                        const double x1 =
                            left + plot_width * static_cast<double>(i) / (count - 1);
                        const double y0 = bottom - plot_height *
                            (curve->values[i - 1] - minimum) / span;
                        const double y1 =
                            bottom - plot_height * (curve->values[i] - minimum) / span;
                        // 超出固定量程的点（比如重捕时云台大角度摆）夹在画布内，
                        // 否则会画到相邻 panel 上去、把别的曲线糊掉。
                        const auto clamp_y = [plot_top, bottom](double value) {
                            return static_cast<int>(
                                std::max(static_cast<double>(plot_top),
                                         std::min(static_cast<double>(bottom), value)));
                        };
                        cv::line(panel,
                                 cv::Point(static_cast<int>(x0), clamp_y(y0)),
                                 cv::Point(static_cast<int>(x1), clamp_y(y1)),
                                 curve->color, 1, cv::LINE_AA);
                    }
                }
                // 图例：每条曲线单独一行（以前两条曲线把字画在同一坐标上 → 叠字）。
                cv::line(panel, cv::Point(margin, legend_y - 4),
                         cv::Point(margin + 14, legend_y - 4), curve->color, 2);
                if (count > 0 && std::isfinite(curve->values.back())) {
                    std::snprintf(text, sizeof(text), "%s  %.3f", curve->name.c_str(),
                                  curve->values.back());
                } else {
                    std::snprintf(text, sizeof(text), "%s  --", curve->name.c_str());
                }
                cv::putText(panel, text, cv::Point(margin + 20, legend_y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.38, curve->color, 1);
                legend_y += 16;
            }
        }

        std::size_t history_ = 300;
        std::size_t active_ = 0;
        int panelsUsed_ = 0;
        std::vector<Curve> curves_;
        std::ofstream csv_;
    };
} // namespace auto_aim::visualization

#endif // ULTRA_VISION_AUTO_AIM_VISUALIZATION_CURVE_PLOTTER_HPP
