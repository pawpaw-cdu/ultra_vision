// 离线：从灯条端点检测 CSV（每行一块板 4 角 [TL,TR,BR,BL]）里按"最近邻质心"
// 关联出单板弧段，喂进模式 A 的 LocalTrajectoryFitter（α/R 固定），打印闭环结果。
// 用法： local_fit_from_csv <dets.csv> <fx fy cx cy> <alpha_deg> <R> <L> <W>
#include "estimation/local_trajectory_fitter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    struct Det { double t; double x[4], y[4]; };
    double cxx(const Det& d) { return (d.x[0]+d.x[1]+d.x[2]+d.x[3])/4; }
    double cyy(const Det& d) { return (d.y[0]+d.y[1]+d.y[2]+d.y[3])/4; }
}

int main(int argc, char** argv)
{
    if (argc < 9) { std::cerr << "usage: local_fit_from_csv <dets.csv> <fx fy cx cy> <alpha_deg> <R> <L> <W>\n"; return 2; }
    auto_aim::LocalTrajectoryFitConfig cfg;
    cfg.K.fx = std::atof(argv[2]); cfg.K.fy = std::atof(argv[3]);
    cfg.K.cx = std::atof(argv[4]); cfg.K.cy = std::atof(argv[5]);
    cfg.omega_init = 4.0;
    const double alpha = std::atof(argv[6]) * M_PI / 180.0;
    const double R = std::atof(argv[7]);
    const double L = std::atof(argv[8]);
    const double W = std::atof(argv[9]);
    cfg.lightbar_len = L; cfg.plate_width = W; cfg.gimbal_pitch = 0.0;

    std::ifstream f(argv[1]); std::string line; std::getline(f, line);
    std::vector<std::string> cols; { std::stringstream ss(line); std::string s; while (std::getline(ss, s, ',')) cols.push_back(s); }
    auto idx = [&](const std::string& n){ for (size_t i=0;i<cols.size();++i) if (cols[i]==n) return (int)i; return -1; };
    const int it=idx("t"), ifr=idx("frame");
    int ic[8]; const char* kn[8]={"x0","y0","x1","y1","x2","y2","x3","y3"};
    for (int i=0;i<8;i++) ic[i]=idx(kn[i]);

    std::vector<Det> dets;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string s; std::vector<double> v;
        while (std::getline(ss, s, ',')) { try { v.push_back(std::stod(s)); } catch (...) { v.push_back(std::nan("")); } }
        if ((int)v.size()<=*std::max_element(ic,ic+8)) continue;
        Det d; d.t=v[it]; for(int i=0;i<4;i++){ d.x[i]=v[ic[2*i]]; d.y[i]=v[ic[2*i+1]]; } dets.push_back(d);
    }
    // 最近邻关联：从最"正对"(最宽)的一块开始，逐帧选质心最近者
    std::sort(dets.begin(), dets.end(), [](const Det&a,const Det&b){return a.t<b.t;});
    std::vector<Det> track; double px=std::nan(""), py=std::nan(""); double pdx=0,pdy=0; bool have=false;
    for (const auto& d : dets) {
        if (!have) { track.push_back(d); px=cxx(d); py=cyy(d); have=true; continue; }
        const double dx=cxx(d)-px, dy=cyy(d)-py;
        if (std::hypot(dx,dy) < 120.0) { track.push_back(d); px=cxx(d); py=cyy(d); pdx=dx; pdy=dy; }
    }
    std::cout << "track len=" << track.size() << " (of " << dets.size() << " dets)\n";
    if (track.size() < 6) { std::cout << "too few\n"; return 1; }

    // 取最近 N 帧窗口
    const int N = std::min<int>(20, track.size());
    std::vector<auto_aim::LightBarObservation> obs;
    for (int i=(int)track.size()-N; i<(int)track.size(); ++i) {
        const auto& d=track[i];
        auto_aim::LightBarObservation a,b;
        a.timestamp=b.timestamp=d.t; a.armor_id=b.armor_id=0;
        a.u_top=d.x[0]; a.v_top=d.y[0]; a.u_bot=d.x[3]; a.v_bot=d.y[3]; a.bar_offset=+0.5*W; // TL->BL
        b.u_top=d.x[1]; b.v_top=d.y[1]; b.u_bot=d.x[2]; b.v_bot=d.y[2]; b.bar_offset=-0.5*W; // TR->BR
        obs.push_back(a); obs.push_back(b);
    }
    auto_aim::LocalTrajectoryFitter fitter(cfg);
    auto_aim::LocalFitResult r;
    const bool ok = fitter.fit(obs, alpha, R, r);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "valid=" << ok << " rmse=" << r.reprojection_rmse_px << "px yaw=" << r.yaw
              << "rad omega=" << r.omega << "rad/s center=" << r.center[0] << "," << r.center[1]
              << "," << r.center[2] << " nbars=" << obs.size() << "\n";
    return 0;
}
