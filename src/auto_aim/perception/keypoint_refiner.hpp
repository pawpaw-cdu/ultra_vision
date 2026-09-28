#ifndef ULTRA_VISION_KEYPOINT_REFINER_HPP
#define ULTRA_VISION_KEYPOINT_REFINER_HPP

// Local refinement of the network's four armor corners.
//
// The neural detector is good at finding armors and classifying them, but its
// corner regression is only as precise as the feature map allows, and the
// measured accuracy report showed the residual error is lateral (the range is
// accurate to ~3%), i.e. corner placement rather than geometry. Teams commonly
// handle this by refining the network's corners with a classical step, which is
// what this does:
//
//   1. take the network's quad as a prior and crop a small ROI around it,
//   2. re-detect the two light bars inside that ROI (threshold + contour +
//      minAreaRect), which is where the classical pipeline is precise but
//      recall-poor -- the network supplies the recall instead,
//   3. rebuild the four corners from the refined bar endpoints,
//   4. accept only if the result stays geometrically consistent (bars roughly
//      parallel and similar height, plate center not jumping), otherwise keep
//      the network corners.

#include <array>
#include <vector>

#include <opencv2/opencv.hpp>

namespace auto_aim
{
    struct KeypointRefinerConfig {
        bool enabled = true;
        // ROI grown around the network quad, as a fraction of its size.
        double margin_ratio = 0.15;
        int binary_threshold = 150;
        // The ROI contains mostly the armor plate and its bars, so Otsu adapts
        // to how dim the bars are (a grazing view can be far darker than a
        // frontal one). Fixed thresholds are still tried as a fallback.
        bool use_otsu = true;
        int morph_size = 3;
        // Accepted light-bar geometry inside the ROI.
        double min_bar_ratio = 0.02;
        double max_bar_ratio = 0.30;
        double max_bar_angle = 35.0;     // degrees from vertical
        double max_height_mismatch = 0.5; // |h1-h2| / max(h1,h2)
        double max_pair_angle_sum = 30.0; // |tilt1 + tilt2| in degrees
        // Reject a refinement that moves the plate center by more than this
        // fraction of the plate width: that means the ROI latched onto
        // something other than the detected armor.
        double max_center_shift_ratio = 0.08;
    };

    struct RefineResult {
        bool refined = false;
        // top-left, top-right, bottom-right, bottom-left
        std::array<cv::Point2f, 4> keypoints{};
        double center_shift_px = 0.0;
        double bar_height_px = 0.0;
    };

    class KeypointRefiner
    {
    public:
        explicit KeypointRefiner(const KeypointRefinerConfig& config = {});

        RefineResult refine(const cv::Mat& bgr,
                            const std::array<cv::Point2f, 4>& nn_keypoints) const;

    private:
        KeypointRefinerConfig config_;
    };
} // namespace auto_aim

#endif // ULTRA_VISION_KEYPOINT_REFINER_HPP
