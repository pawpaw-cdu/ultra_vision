#ifndef ULTRA_VISION_DATASET_HPP
#define ULTRA_VISION_DATASET_HPP

// 实车数据集：把真机每一帧的**图像 + 该帧输入（姿态/弹速/模式）+ 我们的输出**落盘，
// 之后可以在离线环境里改代码/改配置重新跑同一套流水线（tools/replay_dataset），
// 用来"复现问题 / 确认修复有效"，而不必每次都去实车现场。
//
// 目录结构：
//   <dir>/meta.csv        每帧一行（见 DatasetRow 的字段顺序）
//   <dir>/000123.jpg      与 meta.csv 同行号的图像（JPEG，质量可调）
//
// 只写必要的字段、用 CSV 而不是二进制，是为了能用 Excel/python 直接看和筛。

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace auto_aim
{
    struct DatasetRow
    {
        uint64_t frame = 0;
        double t = 0.0;                 // 单调时钟秒（录制起点为 0）
        // ---- 输入：下位机**原始**回传（没做 yaw_sign 之类修正），回放时按配置应用 ----
        double yaw = 0.0;
        double pitch = 0.0;
        double yaw_vel = 0.0;
        double pitch_vel = 0.0;
        double bullet_speed = 0.0;
        int mode = 0;
        // ---- 输出：录制时的流水线结果，回放时可逐项对比 ----
        int armors = 0;
        int pnp = 0;
        int tracker_state = 0;          // 0=LOST 1=TEMP_LOST 2=TRACKING
        int plate_id = -1;
        double aim_yaw = 0.0;
        double aim_pitch = 0.0;
        double aim_yaw_error = 0.0;
        double aim_pitch_error = 0.0;
        int fire = 0;
    };

    inline const char* datasetCsvHeader()
    {
        return "frame,t,yaw,pitch,yaw_vel,pitch_vel,bullet_speed,mode,armors,pnp,tracker_state,"
               "plate_id,aim_yaw,aim_pitch,aim_yaw_error,aim_pitch_error,fire";
    }

    /// @brief 录制器。every_n=每 N 帧录一帧（相机 90+ fps，全录太占地方，默认 3 ≈ 30 fps）。
    class DatasetWriter
    {
    public:
        bool open(const std::string& dir, int every_n, int max_frames, int jpeg_quality)
        {
            directory_ = dir;
            every_n_ = std::max(1, every_n);
            max_frames_ = max_frames;
            jpeg_quality_ = jpeg_quality;
            std::system(("mkdir -p " + directory_).c_str());
            csv_.open(directory_ + "/meta.csv", std::ios::out | std::ios::trunc);
            if (!csv_) return false;
            csv_ << datasetCsvHeader() << '\n';
            return true;
        }

        /// @return 是否真的写了这一帧（没到采样间隔/超过上限时返回 false）
        bool write(const DatasetRow& row, const cv::Mat& image)
        {
            if (!csv_) return false;
            if (max_frames_ > 0 && written_ >= max_frames_) return false;
            if (seen_++ % every_n_ != 0) return false;
            std::ostringstream name;
            name << directory_ << "/" << std::setw(6) << std::setfill('0') << written_ << ".jpg";
            std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
            if (!cv::imwrite(name.str(), image, params)) return false;
            csv_ << row.frame << ',' << std::setprecision(9) << row.t << ',' << row.yaw << ','
                 << row.pitch << ',' << row.yaw_vel << ',' << row.pitch_vel << ',' << row.bullet_speed
                 << ',' << row.mode << ',' << row.armors << ',' << row.pnp << ',' << row.tracker_state
                 << ',' << row.plate_id << ',' << row.aim_yaw << ',' << row.aim_pitch << ','
                 << row.aim_yaw_error << ',' << row.aim_pitch_error << ',' << row.fire << '\n';
            ++written_;
            return true;
        }

        void close()
        {
            if (csv_) csv_.flush();
            csv_.close();
        }

        int written() const { return written_; }
        uint64_t seen() const { return seen_; }
        const std::string& directory() const { return directory_; }

    private:
        std::string directory_;
        std::ofstream csv_;
        int every_n_ = 3;
        int max_frames_ = 0;
        int jpeg_quality_ = 85;
        int written_ = 0;
        uint64_t seen_ = 0;
    };

    /// @brief 读回 meta.csv（图像按行号对应 {i}.jpg）。
    inline bool readDatasetRows(const std::string& dir, std::vector<DatasetRow>& rows)
    {
        std::ifstream csv(dir + "/meta.csv");
        if (!csv) return false;
        std::string line;
        std::getline(csv, line);   // 表头
        while (std::getline(csv, line)) {
            if (line.empty()) continue;
            std::istringstream stream(line);
            std::string field;
            std::vector<std::string> fields;
            while (std::getline(stream, field, ',')) fields.push_back(field);
            if (fields.size() < 17) continue;
            DatasetRow row;
            try {
                row.frame = std::stoull(fields[0]);
                row.t = std::stod(fields[1]);
                row.yaw = std::stod(fields[2]);
                row.pitch = std::stod(fields[3]);
                row.yaw_vel = std::stod(fields[4]);
                row.pitch_vel = std::stod(fields[5]);
                row.bullet_speed = std::stod(fields[6]);
                row.mode = std::stoi(fields[7]);
                row.armors = std::stoi(fields[8]);
                row.pnp = std::stoi(fields[9]);
                row.tracker_state = std::stoi(fields[10]);
                row.plate_id = std::stoi(fields[11]);
                row.aim_yaw = std::stod(fields[12]);
                row.aim_pitch = std::stod(fields[13]);
                row.aim_yaw_error = std::stod(fields[14]);
                row.aim_pitch_error = std::stod(fields[15]);
                row.fire = std::stoi(fields[16]);
            } catch (const std::exception&) {
                continue;
            }
            rows.push_back(row);
        }
        return !rows.empty();
    }
}  // namespace auto_aim

#endif  // ULTRA_VISION_DATASET_HPP
