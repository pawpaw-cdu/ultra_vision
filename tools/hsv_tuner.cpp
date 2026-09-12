#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

namespace {

struct HsvBounds {
    int h_min;
    int h_max;
    int s_min;
    int s_max;
    int v_min;
    int v_max;
};

constexpr const char* kMainWindow = "HSV Tuner";
constexpr const char* kMaskWindow = "Mask";

void onTrackbar(int, void*) {}

void printUsage(std::ostream& out) {
    out << "Usage: hsv_tuner [video_path]\n"
        << "  Without video_path, camera 0 is used.\n"
        << "  Press 'e' to print the current HSV ranges and exit.\n";
}

void printRange(const HsvBounds& range, const std::string& indent,
                const std::string& name) {
    std::cout << indent << name << ":\n"
              << indent << "  h_min: " << range.h_min << "\n"
              << indent << "  h_max: " << range.h_max << "\n"
              << indent << "  s_min: " << range.s_min << "\n"
              << indent << "  s_max: " << range.s_max << "\n"
              << indent << "  v_min: " << range.v_min << "\n"
              << indent << "  v_max: " << range.v_max << "\n";
}

void printInlineRange(const HsvBounds& range) {
    std::cout << "{h_min: " << range.h_min
              << ", h_max: " << range.h_max
              << ", s_min: " << range.s_min
              << ", v_min: " << range.v_min << "}\n";
}

void printParameters(const HsvBounds& range1, const HsvBounds& range2,
                     const HsvBounds& blue_range) {
    std::cout << "\n[HSV Tuner] current parameters\n"
              << "Full block for configs/buff.yaml:\n"
              << "hsv:\n";
    printRange(range1, "  ", "red_range1");
    printRange(range2, "  ", "red_range2");
    printRange(blue_range, "  ", "blue_range");

    std::cout << "\nInline block for src/auto_buff/config/detector.yaml:\n"
              << "red_range1: ";
    printInlineRange(range1);
    std::cout << "red_range2: ";
    printInlineRange(range2);
    std::cout << "blue_range: ";
    printInlineRange(blue_range);
}

cv::Mat rangeMask(const cv::Mat& hsv, const HsvBounds& range) {
    cv::Mat mask;
    cv::inRange(hsv,
                cv::Scalar(range.h_min, range.s_min, range.v_min),
                cv::Scalar(range.h_max, range.s_max, range.v_max),
                mask);
    return mask;
}

cv::Mat makeMask(const cv::Mat& hsv, const HsvBounds& range1,
                 const HsvBounds& range2, const HsvBounds& blue_range,
                 int show_red, int enable_range2, int show_blue) {
    cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
    if (show_red) {
        cv::Mat red_mask = rangeMask(hsv, range1);
        if (enable_range2) {
            cv::bitwise_or(red_mask, rangeMask(hsv, range2), red_mask);
        }
        cv::bitwise_or(mask, red_mask, mask);
    }
    if (show_blue) {
        cv::bitwise_or(mask, rangeMask(hsv, blue_range), mask);
    }
    return mask;
}

void createTrackbar(const char* window, const char* name, int max_value,
                    int value) {
    cv::createTrackbar(name, window, nullptr, max_value, onTrackbar);
    cv::setTrackbarPos(name, window, value);
}

void createTrackbars(const char* window, const HsvBounds& range1,
                     const HsvBounds& range2, const HsvBounds& blue_range,
                     int show_red, int enable_range2, int show_blue) {
    createTrackbar(window, "Show Red", 1, show_red);
    createTrackbar(window, "Red1 H min", 180, range1.h_min);
    createTrackbar(window, "Red1 H max", 180, range1.h_max);
    createTrackbar(window, "Red1 S min", 255, range1.s_min);
    createTrackbar(window, "Red1 S max", 255, range1.s_max);
    createTrackbar(window, "Red1 V min", 255, range1.v_min);
    createTrackbar(window, "Red1 V max", 255, range1.v_max);

    createTrackbar(window, "Enable Red Range2", 1, enable_range2);
    createTrackbar(window, "Red2 H min", 180, range2.h_min);
    createTrackbar(window, "Red2 H max", 180, range2.h_max);
    createTrackbar(window, "Red2 S min", 255, range2.s_min);
    createTrackbar(window, "Red2 S max", 255, range2.s_max);
    createTrackbar(window, "Red2 V min", 255, range2.v_min);
    createTrackbar(window, "Red2 V max", 255, range2.v_max);

    createTrackbar(window, "Show Blue", 1, show_blue);
    createTrackbar(window, "Blue H min", 180, blue_range.h_min);
    createTrackbar(window, "Blue H max", 180, blue_range.h_max);
    createTrackbar(window, "Blue S min", 255, blue_range.s_min);
    createTrackbar(window, "Blue S max", 255, blue_range.s_max);
    createTrackbar(window, "Blue V min", 255, blue_range.v_min);
    createTrackbar(window, "Blue V max", 255, blue_range.v_max);
}

void readTrackbars(const char* window, HsvBounds& range1,
                   HsvBounds& range2, HsvBounds& blue_range,
                   int* show_red, int* enable_range2, int* show_blue) {
    *show_red = cv::getTrackbarPos("Show Red", window);
    range1.h_min = cv::getTrackbarPos("Red1 H min", window);
    range1.h_max = cv::getTrackbarPos("Red1 H max", window);
    range1.s_min = cv::getTrackbarPos("Red1 S min", window);
    range1.s_max = cv::getTrackbarPos("Red1 S max", window);
    range1.v_min = cv::getTrackbarPos("Red1 V min", window);
    range1.v_max = cv::getTrackbarPos("Red1 V max", window);

    *enable_range2 = cv::getTrackbarPos("Enable Red Range2", window);
    range2.h_min = cv::getTrackbarPos("Red2 H min", window);
    range2.h_max = cv::getTrackbarPos("Red2 H max", window);
    range2.s_min = cv::getTrackbarPos("Red2 S min", window);
    range2.s_max = cv::getTrackbarPos("Red2 S max", window);
    range2.v_min = cv::getTrackbarPos("Red2 V min", window);
    range2.v_max = cv::getTrackbarPos("Red2 V max", window);

    *show_blue = cv::getTrackbarPos("Show Blue", window);
    blue_range.h_min = cv::getTrackbarPos("Blue H min", window);
    blue_range.h_max = cv::getTrackbarPos("Blue H max", window);
    blue_range.s_min = cv::getTrackbarPos("Blue S min", window);
    blue_range.s_max = cv::getTrackbarPos("Blue S max", window);
    blue_range.v_min = cv::getTrackbarPos("Blue V min", window);
    blue_range.v_max = cv::getTrackbarPos("Blue V max", window);
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc > 1 &&
        (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(std::cout);
        return 0;
    }

    cv::VideoCapture cap;
    if (argc > 1) {
        cap.open(argv[1]);
    } else {
        cap.open(0);
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open video source";
        if (argc > 1) {
            std::cerr << ": " << argv[1];
        } else {
            std::cerr << ": camera 0";
        }
        std::cerr << std::endl;
        return 1;
    }

    HsvBounds range1{0, 40, 100, 255, 100, 255};
    HsvBounds range2{155, 180, 100, 255, 100, 255};
    HsvBounds blue_range{105, 118, 100, 255, 100, 255};
    int show_red = 1;
    int enable_range2 = 1;
    int show_blue = 1;

    cv::namedWindow(kMainWindow, cv::WINDOW_NORMAL);
    cv::namedWindow(kMaskWindow, cv::WINDOW_NORMAL);
    createTrackbars(kMainWindow, range1, range2, blue_range,
                    show_red, enable_range2, show_blue);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Video source ended or no frame was captured." << std::endl;
            break;
        }

        readTrackbars(kMainWindow, range1, range2, blue_range,
                      &show_red, &enable_range2, &show_blue);

        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask = makeMask(hsv, range1, range2, blue_range,
                                show_red, enable_range2, show_blue);

        cv::Mat overlay = frame.clone();
        overlay.setTo(cv::Scalar(0, 255, 0), mask);
        cv::addWeighted(frame, 0.6, overlay, 0.4, 0.0, overlay);

        cv::imshow(kMainWindow, overlay);
        cv::imshow(kMaskWindow, mask);

        int key = cv::waitKey(1);
        if (key == 'e') {
            readTrackbars(kMainWindow, range1, range2, blue_range,
                          &show_red, &enable_range2, &show_blue);
            printParameters(range1, range2, blue_range);
            break;
        }
        if (key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
