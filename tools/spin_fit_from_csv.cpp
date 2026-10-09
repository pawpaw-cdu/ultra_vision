// 离线：把一条"单块装甲板两根灯条"的端点 CSV 喂给 SpinGeometryEstimator，
// 打印几何解（中心 / 半径 R / 灯条倾角 α / 轴 / 弧覆盖 / 重投影），并把
// "观测 vs 拟合" 端点导出，供绘图与判读。
//
// 用法:
//   spin_fit_from_csv <bars.csv> [fx] [cx] [cy] [c] [L] [out_prefix] [dist_hint]
//   bars.csv 列: t,u_lt,v_lt,u_lb,v_lb,u_rt,v_rt,u_rb,v_rb   （与 spin_video_demo 输出一致）
//   默认对每帧自动判断左右（按 u 均值），保证同一根灯条在整段保持同一 side 号。
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "estimation/spin_geometry_estimator.hpp"

namespace
{
    struct Row
    {
        double t = 0;
        double ul = 0, vl = 0, ulb = 0, vlb = 0;   // 左灯条 上/下
        double ur = 0, vr = 0, urb = 0, vrb = 0;   // 右灯条 上/下
    };
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: spin_fit_from_csv <bars.csv> [fx] [cx] [cy] [c] [L] "
                     "[out_prefix]\n";
        return 2;
    }
    const std::string csv = argv[1];
    double fx = (argc > 2) ? std::atof(argv[2]) : 665.28;
    double cx = (argc > 3) ? std::atof(argv[3]) : 432.0;
    double cy = (argc > 4) ? std::atof(argv[4]) : 960.0;
    double c_off = (argc > 5) ? std::atof(argv[5]) : 0.0655;
    double L_len = (argc > 6) ? std::atof(argv[6]) : 0.060;
    const std::string prefix = (argc > 7) ? argv[7] : "fit";
    const double dist_hint = (argc > 8) ? std::atof(argv[8]) : 0.8;

    std::ifstream is(csv);
    if (!is) { std::cerr << "cannot open " << csv << "\n"; return 1; }
    std::string line;
    std::getline(is, line);   // header
    std::vector<Row> rows;
    while (std::getline(is, line))
    {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<double> v;
        while (std::getline(ss, tok, ','))
        {
            try { v.push_back(std::stod(tok)); } catch (...) { v.push_back(0.0); }
        }
        if (v.size() < 9) continue;
        rows.push_back({v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]});
    }
    std::cout << "rows=" << rows.size() << "  K: fx=fy=" << fx << " cx=" << cx
              << " cy=" << cy << "  c=" << c_off << " L=" << L_len << "\n";
    if (rows.size() < 6) { std::cerr << "too few rows\n"; return 1; }

    std::vector<auto_aim::SpinBarObservation> obs;
    obs.reserve(rows.size() * 2);
    // side 语义：CSV 的 u_lt/u_lb 列 = 这块装甲板"自己的左灯条"（检测器/仿真导出的语义左右），
    // 直接固定映射到 side=-1；u_rt/u_rb → side=+1。**不要按图像 u 重新归一化**——
    // 板绕一圈时语义左灯条会在图像上跑到右边，按图像重排会把 side 标签撕成两半，拟合必然发散。
    for (const auto& r : rows)
    {
        auto_aim::SpinBarObservation a, b;
        a.t = r.t; a.plate = 0; a.side = -1;
        b.t = r.t; b.plate = 0; b.side = +1;
        a.u_top = r.ul;  a.v_top = r.vl;  a.u_bot = r.ulb; a.v_bot = r.vlb;
        b.u_top = r.ur;  b.v_top = r.vr;  b.u_bot = r.urb; b.v_bot = r.vrb;
        obs.push_back(a);
        obs.push_back(b);
    }

    auto_aim::SpinGeometryConfig cfg;
    cfg.K.fx = fx; cfg.K.fy = fx; cfg.K.cx = cx; cfg.K.cy = cy;
    cfg.bar_offset_c = c_off;
    cfg.bar_length_L = L_len;
    cfg.distance_hint = dist_hint;
    cfg.estimate_axis = true;
    cfg.spline_knots = 12;

    auto_aim::SpinGeometryEstimator est(cfg);
    const bool ok = est.fit(obs);
    const auto& res = est.result();
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "fit=" << (ok ? 1 : 0) << " valid=" << (res.valid ? 1 : 0) << "\n";
    std::cout << "center(m) = [" << res.center.x() << ", " << res.center.y() << ", "
              << res.center.z() << "]\n";
    std::cout << "radius R = " << res.radius << " m\n";
    std::cout << "bar tilt alpha = " << res.pitch * 180.0 / M_PI << " deg\n";
    std::cout << "axis = normalize([" << res.axis_tx << ", 1, " << res.axis_tz << "])\n";
    std::cout << "arc coverage = " << res.arc_coverage_deg << " deg\n";
    std::cout << "reproj rmse = " << res.reproj_rmse_px << " px\n";
    std::cout << "t window = [" << res.t_begin << ", " << res.t_end << "]\n";
    if (res.valid && !res.spline.empty())
    {
        const double y0 = res.yawAt(res.t_begin) * 180.0 / M_PI;
        const double y1 = res.yawAt(res.t_end) * 180.0 / M_PI;
        std::cout << "yaw sweep = " << (y1 - y0) << " deg  (y0=" << y0 << " y1=" << y1
                  << ", 不可分辨 k*90)\n";
    }

    // 导出观测 vs 拟合端点
    std::ofstream os(prefix + "_pts.csv");
    os << "t,kind,bar,u,v\n";
    os << std::fixed << std::setprecision(4);
    auto project = [&](const Eigen::Vector3d& p) {
        return std::array<double, 2>{fx * p.x() / p.z() + cx, fx * p.y() / p.z() + cy};
    };
    // 观测
    for (const auto& o : obs)
    {
        os << o.t << ",obs," << (o.side < 0 ? "L" : "R") << "," << o.u_top << "," << o.v_top << "\n";
        os << o.t << ",obs," << (o.side < 0 ? "L" : "R") << "," << o.u_bot << "," << o.v_bot << "\n";
    }
    if (res.valid && !res.spline.empty())
    {
        for (const auto& r : rows)
        {
            const double th = res.yawAt(r.t);
            const auto bars = est.allBars(th);
            if (bars.size() >= 2)
            {
                // allBars 返回 4 板 × 2 灯条；这里只画第 0 块板的两根
                for (int s = 0; s < 2; ++s)
                {
                    const auto& e = bars[s];
                    if (e[0].z() <= 0.05 || e[1].z() <= 0.05) continue;
                    const auto pt = project(e[0]);
                    const auto pb = project(e[1]);
                    os << r.t << ",fit," << (s == 0 ? "L" : "R") << "," << pt[0] << "," << pt[1] << "\n";
                    os << r.t << ",fit," << (s == 0 ? "L" : "R") << "," << pb[0] << "," << pb[1] << "\n";
                }
            }
        }
    }
    std::cout << "wrote " << prefix << "_pts.csv\n";
    return 0;
}
