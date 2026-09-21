#ifndef AUTO_AIM_ENERGY_RUNE_MODEL_HPP
#define AUTO_AIM_ENERGY_RUNE_MODEL_HPP

// OpenVINO inference for the energy-rune keypoint model.
//
// The network is the one already trained and exported by sp_vision_25
// (`assets/yolo11_buff_int8.xml`, single class "fanblade"), so nothing has to
// be retrained. It outputs, per anchor:
//     [cx, cy, w, h, score, 6 * (x, y)]   -> shape [1, 17, 8400]
// where the six keypoints are the four plate corners, the plate centre and a
// point on the arm. `RuneDetector` turns them into fan blades and the rune
// centre.

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace auto_aim::energy
{
    struct RuneModelConfig
    {
        std::string model_path;
        std::string device = "CPU";
        int input_size = 640;
        int input_height = 0;             // 0 = 与 input_size 相同（方形输入）
        // 输出布局："sp_vision_6kpt"（单类 + 6 点×2）或
        // "v8_pose_5kpt"（N 类分数 + 5 点×3，深大 RM2026 五点模型）。
        std::string output_layout = "sp_vision_6kpt";
        int num_classes = 1;
        int num_keypoints = 6;
        bool keypoints_have_confidence = false;
        bool class_scores_are_logits = false;
        float keypoint_confidence_threshold = 0.8f;
        int min_valid_keypoints = 3;
        float confidence_threshold = 0.7f;
        float nms_threshold = 0.4f;      // 旧布局：按框 IoU 抑制
        // 深大五点模型按"中心点距离"抑制重复框（其 infer 参考实现用 30 px），
        // 相邻扇叶的框重叠很明显，用 IoU 会把大符同时点亮的两片合成一片。
        float nms_center_dist_px = 30.0f;
        // 靶面点相对靶心的最小像素半径。网络在离域画面上会"塌缩"式误检
        // （5 个点挤在 2~5 px 内），几何上不可能是 0.16 m 的靶面；真实靶面
        // 在 15 m 处仍有 4 px 以上，所以 3 px 只用来剔除退化框。
        float min_plate_radius_px = 3.0f;
        int pad_value = 114;            // matches the model's export metadata
        // The keypoint model only fires when the rune occupies roughly the
        // share of the frame it was trained on (~1/3). Padding the incoming
        // image into a larger grey canvas shrinks it inside the network's
        // input without resampling the pixels, so the PnP precision of a
        // closer camera is kept while detection keeps working.
        // <1 表示"数字变焦"：中心裁剪到 scale 倍大小再 letterbox，仿真里 6 m
        // 处的小符在画面里只有约 1/9 宽（训练尺度约 1/3），放大后才稳定出点。
        double input_pad_scale = 1.0;
        // 送进网络的像素缩放：像素值 × input_scale。
        // 深大五点模型的 /255 已经折进第一层卷积权重（图内没有归一化节点），
        // 必须传 1.0；sp_vision 的旧模型按 0~1 归一化训练，用 1/255。
        double input_scale = 1.0 / 255.0;
        bool reverse_input_channels = true;
        int num_threads = 4;
        std::string performance_mode = "latency";
    };

    class RuneModel
    {
    public:
        struct Object
        {
            cv::Rect_<float> rect;
            float prob = 0.0f;
            int class_id = 0;
            // 靶面中心（图像坐标）：五点布局取四个靶面点的均值，六点布局取
            // 第 5 个关键点（sp_vision 的靶心点）。
            cv::Point2f center{0.0f, 0.0f};
            std::vector<cv::Point2f> keypoints;
            std::vector<float> keypoint_confidence;
        };

        explicit RuneModel(const RuneModelConfig& config);
        ~RuneModel();

        RuneModel(RuneModel&&) noexcept;
        RuneModel& operator=(RuneModel&&) noexcept;

        // All detections above the threshold, NMS filtered.
        std::vector<Object> detect(const cv::Mat& image);
        // Only the highest-score detection (the rune lights one blade per
        // round on the small rune, two on the large one).
        std::vector<Object> detectBest(const cv::Mat& image);

        /// @brief 设置"画布缩放"（= 配置里的 input_pad_scale，逐帧可变）。
        /// 检测器用上一帧量到的轨道半径把它算出来，好让机关在 640 宽的输入里
        /// 始终保持训练尺度（见 rune_detector 的 updateCanvasScale）。
        void setCanvasScale(double scale);

        double latencyMs() const { return latency_ms_; }
        // 本帧所有 anchor 里最高的类别分（不管是否过阈值）：用来区分"网络没看见"
        // 和"看见了但被阈值筛掉"——这是判断该不该动阈值的关键数据。
        float lastBestScore() const { return last_best_score_; }


    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        double latency_ms_ = 0.0;
        float last_best_score_ = 0.0f;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_MODEL_HPP
