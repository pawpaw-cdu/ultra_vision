// 把一段视频里每一帧的**全部**装甲板检测导出成 CSV，供离线做连续跟踪/几何拟合。
// 与 spin_video_demo 的区别：这里不做任何跟踪决策，只忠实落盘原始检测，
// 跟踪/择板策略放到离线脚本里迭代，避免"贪心最近邻"把不同装甲板缝进同一条序列。
//
// 用法:
//   spin_det_dump <video> <model.xml> <out.csv> [max_frames] [score_thres]
// 输出列:
//   frame,t,det,conf,cls,large,is_small,
//   x0,y0,x1,y1,x2,y2,x3,y3          # 原始关键点 tl,tr,br,bl
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "perception/nn_detector.hpp"

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: spin_det_dump <video> <model.xml> <out.csv> "
                     "[max_frames] [score_thres]\n";
        return 2;
    }
    const std::string video = argv[1];
    const std::string model = argv[2];
    const std::string out_csv = argv[3];
    const int max_frames = (argc > 4) ? std::atoi(argv[4]) : 1000000;
    const double score_thres = (argc > 5) ? std::atof(argv[5]) : 0.25;

    cv::VideoCapture cap(video);
    if (!cap.isOpened()) { std::cerr << "cannot open video: " << video << "\n"; return 1; }
    const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!(fps > 1.0)) fps = 30.0;
    std::cout << "video " << W << "x" << H << " @" << fps << "fps\n";

    auto_aim::NnDetectorConfig cfg;
    cfg.model_path = model;
    cfg.device = "CPU";
    cfg.output_format = "rp24";
    cfg.score_threshold = score_thres;
    cfg.min_confidence = score_thres;
    cfg.filter_by_color = false;   // 蓝/红都留，后面按颜色/尺寸筛
    auto_aim::NnArmorDetector detector(cfg);

    std::ofstream os(out_csv);
    os << "frame,t,det,conf,cls,large,is_small,"
          "x0,y0,x1,y1,x2,y2,x3,y3\n";
    os << std::fixed << std::setprecision(3);

    const double dt = 1.0 / fps;
    int frame_idx = 0, written = 0, total_det = 0;
    cv::Mat frame;
    while (cap.read(frame) && frame_idx < max_frames)
    {
        const auto dets = detector.detect(frame);
        const double t = frame_idx * dt;
        int k = 0;
        for (const auto& d : dets)
        {
            if (d.keypoints.size() < 4) continue;
            os << frame_idx << ',' << t << ',' << k++ << ','
               << d.confidence << ',' << d.class_id << ','
               << (d.large ? 1 : 0) << ',' << (d.large ? 0 : 1);
            for (const auto& kp : d.keypoints)
                os << ',' << kp.x << ',' << kp.y;
            os << '\n';
            ++total_det;
        }
        ++frame_idx;
        ++written;
    }
    std::cout << "wrote " << out_csv << "  frames=" << written
              << "  detections=" << total_det << "\n";
    return 0;
}
