// 仿真入口：装配 SimRunner，执行检测→估计→瞄准→开火。

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <opencv2/opencv.hpp>

#include "app/config_paths.hpp"
#include "app/sim_runner.hpp"

namespace
{
    const char* trackerStateName(auto_aim::TrackerState state)
    {
        switch (state) {
        case auto_aim::TrackerState::DETECTING: return "DETECTING";
        case auto_aim::TrackerState::TRACKING: return "TRACKING";
        case auto_aim::TrackerState::TEMP_LOST: return "TEMP_LOST";
        case auto_aim::TrackerState::LOST: return "LOST";
        }
        return "UNKNOWN";
    }
}

int main(int argc, char** argv)
{
    auto_aim::app::SimRunner runner(auto_aim::app::resolveConfigDir(argc, argv));
    if (!runner.initialize()) return 1;

    int frames = 0;
    auto fps_start = std::chrono::steady_clock::now();
    while (!runner.exitRequested()) {
        if (runner.testSecondsExpired()) break;
        if (!runner.fetchFrame()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            runner.setKey(1);
            continue;
        }

        runner.prepareFrame();
        runner.runAim();

        ++frames;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
            auto_aim::Tracker& tracker = runner.tracker();
            std::cout << "Processed FPS: " << frames / elapsed
                      << ", lights: " << runner.lightCount()
                      << ", armors: " << runner.armorCount()
                      << ", PnP: " << runner.pnpCount()
                      << ", tracker: " << trackerStateName(tracker.getState())
                      << ", lost: " << tracker.getLostCount()
                      << ", omega: " << tracker.getOmega()
                      << ", fire: " << (tracker.getState() == auto_aim::TrackerState::TRACKING)
                      << std::endl;
            frames = 0;
            fps_start = now;
        }

        if (runner.showDisplay()) {
            cv::imshow("Ultra Vision", runner.image());
            runner.setKey(cv::waitKey(1));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            runner.setKey(1);
        }
    }
    return 0;
}
