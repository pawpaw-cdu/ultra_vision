#ifndef AUTO_AIM_NN_DETECTOR_UTILS_HPP
#define AUTO_AIM_NN_DETECTOR_UTILS_HPP

// 从 perception/nn_detector.cpp 抽出的无状态工具（类别表/letterbox/解码几何）。

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "perception/nn_detector.hpp"

namespace auto_aim
{
namespace nn_detail
{
namespace
{
        // The 38-class head of the SZU model, in output order. Mirrors
        // sp_vision_25's armor_properties table.
        struct ClassSpec {
            NnColor color;
            NnName name;
            bool large;
        };

        const std::array<ClassSpec, 38>& classTable()
        {
            static const std::array<ClassSpec, 38> table{{
                {NnColor::blue, NnName::sentry, false},
                {NnColor::red, NnName::sentry, false},
                {NnColor::extinguish, NnName::sentry, false},
                {NnColor::blue, NnName::one, false},
                {NnColor::red, NnName::one, false},
                {NnColor::extinguish, NnName::one, false},
                {NnColor::blue, NnName::two, false},
                {NnColor::red, NnName::two, false},
                {NnColor::extinguish, NnName::two, false},
                {NnColor::blue, NnName::three, false},
                {NnColor::red, NnName::three, false},
                {NnColor::extinguish, NnName::three, false},
                {NnColor::blue, NnName::four, false},
                {NnColor::red, NnName::four, false},
                {NnColor::extinguish, NnName::four, false},
                {NnColor::blue, NnName::five, false},
                {NnColor::red, NnName::five, false},
                {NnColor::extinguish, NnName::five, false},
                {NnColor::blue, NnName::outpost, false},
                {NnColor::red, NnName::outpost, false},
                {NnColor::extinguish, NnName::outpost, false},
                {NnColor::blue, NnName::base, true},
                {NnColor::red, NnName::base, true},
                {NnColor::extinguish, NnName::base, true},
                {NnColor::purple, NnName::base, true},
                {NnColor::blue, NnName::base, false},
                {NnColor::red, NnName::base, false},
                {NnColor::extinguish, NnName::base, false},
                {NnColor::purple, NnName::base, false},
                {NnColor::blue, NnName::three, true},
                {NnColor::red, NnName::three, true},
                {NnColor::extinguish, NnName::three, true},
                {NnColor::blue, NnName::four, true},
                {NnColor::red, NnName::four, true},
                {NnColor::extinguish, NnName::four, true},
                {NnColor::blue, NnName::five, true},
                {NnColor::red, NnName::five, true},
                {NnColor::extinguish, NnName::five, true},
            }};
            return table;
        }

        const char* colorName(NnColor color)
        {
            switch (color) {
            case NnColor::blue: return "blue";
            case NnColor::red: return "red";
            case NnColor::extinguish: return "extinguish";
            case NnColor::purple: return "purple";
            }
            return "unknown";
        }

        const char* nameName(NnName name)
        {
            switch (name) {
            case NnName::one: return "one";
            case NnName::two: return "two";
            case NnName::three: return "three";
            case NnName::four: return "four";
            case NnName::five: return "five";
            case NnName::sentry: return "sentry";
            case NnName::outpost: return "outpost";
            case NnName::base: return "base";
            case NnName::not_armor: return "not_armor";
            }
            return "unknown";
        }

        double sigmoid(double value)
        {
            return 1.0 / (1.0 + std::exp(-value));
        }

        // RP24 vocabulary. The published description lists the colours as
        // (red, blue, gray, purple), but the reference implementation decodes
        // channel 0 as blue and 1 as red, so follow the code: 0=blue, 1=red,
        // 2=extinguish, 3=purple.
        NnColor rp24Color(int color_id)
        {
            switch (color_id) {
            case 0: return NnColor::blue;
            case 1: return NnColor::red;
            case 2: return NnColor::extinguish;
            default: return NnColor::purple;
            }
        }

        NnName rp24Name(int num_id)
        {
            switch (num_id) {
            case 0: return NnName::sentry;
            case 1: return NnName::one;
            case 2: return NnName::two;
            case 3: return NnName::three;
            case 4: return NnName::four;
            case 5: return NnName::five;
            case 6: return NnName::outpost;
            default: return NnName::base;   // 7 = Bs, 8 = Bb
            }
        }

        // Order the four corners as top-left, top-right, bottom-right,
        // bottom-left, matching both the model's training convention and the
        // point order used by solveArmorPnP().
        void sortKeypoints(std::array<cv::Point2f, 4>& keypoints)
        {
            std::sort(keypoints.begin(), keypoints.end(),
                [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                    return lhs.y < rhs.y;
                });
            if (keypoints[0].x > keypoints[1].x) {
                std::swap(keypoints[0], keypoints[1]);   // top-left, top-right
            }
            if (keypoints[2].x > keypoints[3].x) {
                std::swap(keypoints[2], keypoints[3]);   // bottom-left, bottom-right
            }
            std::swap(keypoints[2], keypoints[3]);       // -> top-right, bottom-right
        }
}
}
}

#endif // AUTO_AIM_NN_DETECTOR_UTILS_HPP
