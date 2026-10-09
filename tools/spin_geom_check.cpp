// 离线复核 SpinGeometryEstimator：读仿真 observation.csv（每帧 4 角点 + 板号 + 时间），
// 转成多灯条观测，跑一次全局几何拟合，打印是否收敛与估出的几何/覆盖。
// 用法： spin_geom_check <observation.csv> [fx fy cx cy c L]
#include "estimation/spin_geometry_estimator.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) { std::cerr << "usage: spin_geom_check <observation.csv> [fx fy cx cy c L]\n"; return 2; }
    auto_aim::SpinGeometryConfig cfg;
    double c = 0.0658, L = 0.060;
    if (argc >= 7) {
        cfg.K.fx = std::atof(argv[2]); cfg.K.fy = std::atof(argv[3]);
        cfg.K.cx = std::atof(argv[4]); cfg.K.cy = std::atof(argv[5]);
        c = std::atof(argv[6]); L = std::atof(argv[7]);
    }
    cfg.bar_offset_c = c;
    cfg.bar_length_L = L;
    cfg.distance_hint = 2.5;
    cfg.estimate_axis = true;

    std::ifstream f(argv[1]);
    if (!f) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }
    std::string line; std::getline(f, line);
    // 列名 → 下标
    std::vector<std::string> cols; { std::stringstream ss(line); std::string t; while (std::getline(ss, t, ',')) cols.push_back(t); }
    auto idx = [&](const std::string& n) { for (size_t i = 0; i < cols.size(); ++i) if (cols[i] == n) return (int)i; return -1; };
    const int i_time = idx("source_time_us") >= 0 ? idx("source_time_us") : idx("time_us");
    const int i_plate = idx("plate_id");
    const int i_u0 = idx("u0"), i_v0 = idx("v0"), i_u1 = idx("u1"), i_v1 = idx("v1");
    const int i_u2 = idx("u2"), i_v2 = idx("v2"), i_u3 = idx("u3"), i_v3 = idx("v3");
    if (i_time < 0 || i_plate < 0 || i_u0 < 0 || i_u1 < 0 || i_u2 < 0 || i_u3 < 0)
    { std::cerr << "missing columns\n"; return 2; }

    std::vector<auto_aim::SpinBarObservation> obs;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string t; std::vector<double> v;
        while (std::getline(ss, t, ',')) { try { v.push_back(std::stod(t)); } catch (...) { v.push_back(std::nan("")); } }
        if ((int)v.size() <= std::max({i_time, i_plate, i_u0, i_v0, i_u1, i_v1, i_u2, i_v2, i_u3, i_v3})) continue;
        const double time_s = v[i_time] * 1e-6;
        const int plate = (int)v[i_plate];
        auto_aim::SpinBarObservation a, b;
        a.t = time_s; a.plate = plate; a.side = +1;
        // observation.csv 的顺序 = Armor::Points_2D = {left.bottom, left.top, right.top, right.bottom}
        a.u_bot = v[i_u0]; a.v_bot = v[i_v0];   // left.bottom
        a.u_top = v[i_u1]; a.v_top = v[i_v1];   // left.top
        b.t = time_s; b.plate = plate; b.side = -1;
        b.u_top = v[i_u2]; b.v_top = v[i_v2];   // right.top
        b.u_bot = v[i_u3]; b.v_bot = v[i_v3];   // right.bottom
        obs.push_back(a); obs.push_back(b);
    }
    std::cout << "bars=" << obs.size() << "\n";

    auto_aim::SpinGeometryEstimator est(cfg);
    const bool ok = est.fit(obs);
    const auto& r = est.result();
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "valid=" << ok << " R=" << r.radius << " pitch_deg=" << r.pitch * 180 / M_PI
              << " axis(tx,tz)=(" << r.axis_tx << "," << r.axis_tz << ")"
              << " center=" << r.center.transpose()
              << " arc_cov_deg=" << r.arc_coverage_deg << " reproj_rmse=" << r.reproj_rmse_px
              << " t=[" << r.t_begin << "," << r.t_end << "]\n";
    return ok ? 0 : 1;
}
