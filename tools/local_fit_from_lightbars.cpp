// 用仿真器 light_bars.csv（每根灯条中心+上下端点）验证模式 A 拟合器：
//   端点(3D相机系) --投影(K)--> 像素 --LocalTrajectoryFitter--> yaw/ω
// 真实数据、无检测噪声；用于判断"拟合器本身能否正常输出强 yaw/ω"。
//
// 观测构造**严格对齐在线链路** src/auto_aim/Kalman/tracker_localfit.cpp：
//   · top/bottom：图像上端(v 更小) = top（拟合器要求 top 是相机 y 更小的一端）；
//   · bar_offset：图像左灯条 = +W/2、右灯条 = -W/2（= online armor.left→+W/2）。
//     **不能用 bar_id 奇偶**：灯条在板面的左右随板号翻转（仿真 mesh 顶点序），
//     只有"正对相机"时图像左右才对应板面切向；
//   · 弧段：取该板"正对相机"（两灯条像素横向间距最大）的 N 帧连续弧，
//     与在线"一块板连续可见弧"同义。
// 用法: local_fit_from_lightbars <csv> <fx fy cx cy> <alpha_deg> <R> <L> <W> <gimbal_pitch_deg> [plate] [frames]
//
// 反向标定用（env，2026-10-12）：
//   ULTRA_VISION_LF_BLUR_MS=<ms>   运动模糊模型：观测端点 = 曝光窗口内 [t-ms, t] 的
//                                  投影位置平均（K=9 个亚采样，端点按相邻帧线性插值）。
//                                  用来回答"实车检测端等效有多少运动误差"。
//   ULTRA_VISION_LF_SHIFT_MS=<ms>  **负对照**：把每个观测的时间戳整体平移 ms
//                                  （即用 t+ms 时刻的端点）。常量时间偏移是**纯相位**误差，
//                                  会被 θ0/ω 吸收 → 残差应当几乎不变。
//   ULTRA_VISION_LF_BLUR_LOG=1     额外打印"等效模糊像素数"（按图像速度×曝光时间算）。
#include "estimation/local_trajectory_fitter.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace
{
    struct BarObs { int bar_id = 0; double cu = 0.0, ut = 0, vt = 0, ub = 0, vb = 0; };
    struct Frame { long long t_us = 0; std::vector<BarObs> bars; };
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 10) { std::cerr << "usage: local_fit_from_lightbars <csv> <fx fy cx cy> <alpha_deg> <R> <L> <W> <gp_deg> [plate] [frames]\n"; return 2; }
    auto_aim::LocalTrajectoryFitConfig cfg;
    cfg.K.fx=std::atof(argv[2]); cfg.K.fy=std::atof(argv[3]); cfg.K.cx=std::atof(argv[4]); cfg.K.cy=std::atof(argv[5]);
    const double alpha=std::atof(argv[6])*M_PI/180.0, R=std::atof(argv[7]);
    const double L=std::atof(argv[8]), W=std::atof(argv[9]);
    cfg.lightbar_len=L; cfg.plate_width=W; cfg.gimbal_pitch=std::atof(argv[10])*M_PI/180.0;
    cfg.omega_init=3.0; cfg.center_init={{(float)0.0,(float)0.0,(float)3.0}};
    const double blur_ms = std::getenv("ULTRA_VISION_LF_BLUR_MS")
        ? std::atof(std::getenv("ULTRA_VISION_LF_BLUR_MS")) : 0.0;
    const double shift_ms = std::getenv("ULTRA_VISION_LF_SHIFT_MS")
        ? std::atof(std::getenv("ULTRA_VISION_LF_SHIFT_MS")) : 0.0;
    const int want_plate = (argc>11)? std::atoi(argv[11]) : 0;
    const int want_frames = (argc>12)? std::atoi(argv[12]) : 20;

    std::ifstream f(argv[1]); std::string line; std::getline(f,line);
    std::vector<std::string> cols; { std::stringstream ss(line); std::string s; while(std::getline(ss,s,',')) cols.push_back(s);}
    auto idx=[&](const std::string&n){for(size_t i=0;i<cols.size();++i) if(cols[i]==n) return (int)i; return -1;};
    const int i_t=idx("time_us"),i_bar=idx("bar_id"),i_pl=idx("plate_id");
    const int i0=idx("x0"),i1=idx("y0"),i2=idx("z0"),i3=idx("x1"),i4=idx("y1"),i5=idx("z1");
    // 像素模式：输入是仿真 observation.csv（检测到的装甲板四角 lb,lt,rt,rb + plate_id + time_us）。
    const int iu0=idx("u0"),iu1=idx("u1"),iu2=idx("u2"),iu3=idx("u3");
    const int iv0=idx("v0"),iv1=idx("v1"),iv2=idx("v2"),iv3=idx("v3");
    const bool pixel_mode = (iu0>=0 && iv0>=0 && iu3>=0 && iv3>=0);
    // 可选：离线按镜头畸变先去畸变（在线拟合器会做这一步；实车对账必须一致）。
    //   ULTRA_VISION_LOCAL_FIT_TOOL_DIST="k1,k2,p1,p2[,k3]"
    std::vector<double> dist_v;
    if (const char* de = std::getenv("ULTRA_VISION_LOCAL_FIT_TOOL_DIST")) {
        std::stringstream ds(de); std::string tok;
        while (std::getline(ds, tok, ',')) { try { dist_v.push_back(std::stod(tok)); } catch (...) {} }
    }
    const cv::Mat Kmat = (cv::Mat_<double>(3, 3) << cfg.K.fx, 0.0, cfg.K.cx,
                          0.0, cfg.K.fy, cfg.K.cy, 0.0, 0.0, 1.0);
    cv::Mat Dmat;
    if (dist_v.size() >= 4) { Dmat = cv::Mat(static_cast<int>(dist_v.size()), 1, CV_64F); for (size_t i=0;i<dist_v.size();++i) Dmat.at<double>((int)i,0)=dist_v[i]; }
    auto und=[&](double& u, double& vv){ if (Dmat.empty()) return;
        std::vector<cv::Point2f> in{cv::Point2f((float)u,(float)vv)}, out;
        cv::undistortPoints(in,out,Kmat,Dmat,cv::noArray(),Kmat);
        if(!out.empty()){u=out[0].x; vv=out[0].y;} };
    auto proj=[&](double X,double Y,double Z,double&u,double&v){u=cfg.K.fx*X/Z+cfg.K.cx; v=cfg.K.fy*Y/Z+cfg.K.cy;};
    const int maxcol = pixel_mode ? std::max({i_t,i_pl,iu0,iv0,iu1,iv1,iu2,iv2,iu3,iv3}) : std::max({i_t,i_pl,i0,i1,i2,i3,i4,i5});

    // ---- 反向标定用：每根灯条的 3D 端点时间序列（用于亚采样插值）-----------------
    // key = bar_id，value = 按时间排序的 {t_us, x0,y0,z0, x1,y1,z1}
    std::map<int, std::vector<std::array<double, 7>>> bar_series;
    // 8 根灯条 × 每帧：这既是"模糊"的采样源，也是算图像速度的源。
    // 先扫一遍文件把 3D 端点收全（pixel 模式下没有 3D 端点，模糊功能自动失效）。
    {
        std::ifstream f2(argv[1]); std::string l2; std::getline(f2, l2);
        if (!pixel_mode && i0 >= 0 && i3 >= 0)
        {
            while (std::getline(f2, l2))
            {
                if (l2.empty()) continue;
                std::stringstream ss(l2); std::string tok; std::vector<double> v;
                while (std::getline(ss, tok, ',')) { try { v.push_back(std::stod(tok)); } catch (...) { v.push_back(std::nan("")); } }
                if ((int)v.size() <= i3) continue;
                if ((int)v[i_pl] != want_plate) continue;
                const int bid = (i_bar >= 0) ? static_cast<int>(v[i_bar]) : 0;
                bar_series[bid].push_back({v[i_t], v[i0], v[i1], v[i2], v[i3], v[i4], v[i5]});
            }
            for (auto& kv : bar_series)
                std::sort(kv.second.begin(), kv.second.end(),
                          [](const std::array<double,7>& a, const std::array<double,7>& b){ return a[0] < b[0]; });
        }
    }
    // 在时间 tau(us) 上线性插值该灯条的两个端点；返回 false 表示没有足够样本。
    auto sample_bar = [&](int bid, double tau_us, double* t3, double* b3) -> bool {
        auto it = bar_series.find(bid);
        if (it == bar_series.end() || it->second.size() < 2) return false;
        const auto& v = it->second;
        if (tau_us <= v.front()[0]) { for (int k=0;k<3;++k){ t3[k]=v.front()[1+k]; b3[k]=v.front()[4+k]; } return true; }
        if (tau_us >= v.back()[0])  { for (int k=0;k<3;++k){ t3[k]=v.back()[1+k];  b3[k]=v.back()[4+k];  } return true; }
        for (std::size_t i = 1; i < v.size(); ++i)
            if (v[i][0] >= tau_us)
            {
                const double w = (tau_us - v[i-1][0]) / std::max(1.0, v[i][0] - v[i-1][0]);
                for (int k=0;k<3;++k) { t3[k] = v[i-1][1+k] + w*(v[i][1+k]-v[i-1][1+k]); b3[k] = v[i-1][4+k] + w*(v[i][4+k]-v[i-1][4+k]); }
                return true;
            }
        return false;
    };

    // 按帧时间分组（文件顺序 == 时间顺序）
    std::vector<Frame> frames; std::map<long long,size_t> frame_idx;
    while(std::getline(f,line)){ if(line.empty())continue;
        std::stringstream ss(line); std::string s; std::vector<double> v;
        while(std::getline(ss,s,',')){ try{v.push_back(std::stod(s));}catch(...){v.push_back(std::nan(""));}}
        if((int)v.size()<=maxcol) continue;
        if((int)v[i_pl]!=want_plate) continue;
        const long long t_us = static_cast<long long>(v[i_t]);
        auto it = frame_idx.find(t_us);
        if (it == frame_idx.end()) { frame_idx[t_us] = frames.size(); frames.push_back(Frame{t_us, {}}); it = frame_idx.find(t_us); }
        if (pixel_mode)
        {
            // 检测到的角点顺序 = Armor::Points_2D = {left.bottom, left.top, right.top, right.bottom}
            double lb_u=v[iu0], lb_v=v[iv0], lt_u=v[iu1], lt_v=v[iv1];
            double rt_u=v[iu2], rt_v=v[iv2], rb_u=v[iu3], rb_v=v[iv3];
            und(lb_u, lb_v); und(lt_u, lt_v); und(rt_u, rt_v); und(rb_u, rb_v);
            BarObs L; L.bar_id=0; L.ub=lb_u; L.vb=lb_v; L.ut=lt_u; L.vt=lt_v; L.cu=0.5*(L.ut+L.ub);
            BarObs R; R.bar_id=1; R.ut=rt_u; R.vt=rt_v; R.ub=rb_u; R.vb=rb_v; R.cu=0.5*(R.ut+R.ub);
            if (std::isnan(L.ut)||std::isnan(R.ut)) continue;
            frames[it->second].bars.push_back(L);
            frames[it->second].bars.push_back(R);
        }
        else
        {
            BarObs bo; bo.bar_id = (i_bar>=0)? static_cast<int>(v[i_bar]) : 0;
            double uu0,vv0,uu1,vv1;
            if (blur_ms > 0.0)
            {
                // **运动模糊模型**：观测端点 = 曝光窗口 [t-blur, t] 内投影位置的平均。
                // K=9 个亚采样（中点法），端点用相邻帧线性插值。真实探测器在模糊条上
                // 报出的角点≈拖影的质心，所以"取平均"是一阶模型。
                const int K = 9;
                double s0u=0,s0v=0,s1u=0,s1v=0; int cnt=0;
                for (int k = 0; k < K; ++k)
                {
                    const double frac = (k + 0.5) / K;                 // (0,1)
                    const double tau = v[i_t] - blur_ms * 1000.0 * (1.0 - frac);
                    double t3[3], b3[3];
                    if (!sample_bar(bo.bar_id, tau, t3, b3)) continue;
                    double au,av,bu,bv; proj(t3[0],t3[1],t3[2],au,av); proj(b3[0],b3[1],b3[2],bu,bv);
                    s0u+=au; s0v+=av; s1u+=bu; s1v+=bv; ++cnt;
                }
                if (cnt == 0) { proj(v[i0],v[i1],v[i2],uu0,vv0); proj(v[i3],v[i4],v[i5],uu1,vv1); }
                else { uu0=s0u/cnt; vv0=s0v/cnt; uu1=s1u/cnt; vv1=s1v/cnt; }
            }
            else
            {
                proj(v[i0],v[i1],v[i2],uu0,vv0); proj(v[i3],v[i4],v[i5],uu1,vv1);
            }
            // top = 图像上端（投影后 v 更小的端点）；(x0,y0,z0)/(x1,y1,z1) 来自网格顶点序，
            // 实测恒有 y0>y1（相机 y 向下）→ x0 端是下端，直接当 top 会造成 ~L 的系统偏差。
            if (shift_ms != 0.0)
            {
                double t3[3], b3[3];
                if (sample_bar(bo.bar_id, v[i_t] + shift_ms * 1000.0, t3, b3))
                { proj(t3[0],t3[1],t3[2],uu0,vv0); proj(b3[0],b3[1],b3[2],uu1,vv1); }
            }
            if (vv1 < vv0) { bo.ut=uu1; bo.vt=vv1; bo.ub=uu0; bo.vb=vv0; }
            else           { bo.ut=uu0; bo.vt=vv0; bo.ub=uu1; bo.vb=vv1; }
            bo.cu = 0.5*(uu0+uu1);
            frames[it->second].bars.push_back(bo);
        }
    }
    std::cout << (pixel_mode? "[pixel/observation.csv mode]" : "[light_bars 3D mode]") << "\n";

    // 只保留两根灯条都在的帧
    std::vector<Frame> ok;
    for (auto& fr : frames) if (fr.bars.size() == 2) ok.push_back(fr);
    std::cout<<"plate "<<want_plate<<" frames="<<ok.size()<<"\n";
    if(ok.size()<6){std::cout<<"too few\n";return 1;}

    // 选"正对相机"的弧段：两灯条像素横向间距最大的 N 帧连续窗口
    const int N = std::max(4, std::min<int>(want_frames, (int)ok.size()));
    auto sep=[&](const Frame& fr){ return std::abs(fr.bars[0].cu - fr.bars[1].cu); };
    int best_start = 0; double best_score = -1.0;
    for (int s = 0; s + N <= (int)ok.size(); ++s)
    {
        double score = 0.0; for (int k = 0; k < N; ++k) score += sep(ok[s+k]);
        if (score > best_score) { best_score = score; best_start = s; }
    }

    std::vector<auto_aim::LightBarObservation> win;
    for (int k = 0; k < N; ++k)
    {
        const Frame& fr = ok[best_start + k];
        const BarObs* left  = fr.bars[0].cu <= fr.bars[1].cu ? &fr.bars[0] : &fr.bars[1];
        const BarObs* right = fr.bars[0].cu <= fr.bars[1].cu ? &fr.bars[1] : &fr.bars[0];
        auto push=[&](const BarObs* b, double off){
            auto_aim::LightBarObservation o; o.timestamp=fr.t_us*1e-6; o.armor_id=want_plate;
            o.u_top=b->ut; o.v_top=b->vt; o.u_bot=b->ub; o.v_bot=b->vb; o.bar_offset=off;
            win.push_back(o);
        };
        push(left, +0.5*W);   // 图像左灯条 = +W/2（同 online armor.left）
        push(right, -0.5*W);  // 图像右灯条 = -W/2
    }
    auto_aim::LocalTrajectoryFitter fitter(cfg);
    auto_aim::LocalFitResult r;
    const bool fit_ok=fitter.fit(win,alpha,R,r);
    std::cout<<std::fixed<<std::setprecision(4);
    // 等效模糊像素数：|图像速度| × 曝光时间（用弧段首尾灯条中心的位移估速度）
    if (blur_ms > 0.0 || shift_ms != 0.0)
    {
        const Frame& f0 = ok[best_start];
        const Frame& f1 = ok[best_start + N - 1];
        const double dt = (f1.t_us - f0.t_us) * 1e-6;
        if (dt > 1e-6)
        {
            const double d0 = std::hypot(f0.bars[0].cu - f1.bars[0].cu, f0.bars[0].vt - f1.bars[0].vt);
            const double d1 = std::hypot(f0.bars[1].cu - f1.bars[1].cu, f0.bars[1].vt - f1.bars[1].vt);
            const double speed = 0.5 * (d0 + d1) / dt;   // px/s
            std::cout << "  [注入] blur=" << blur_ms << "ms shift=" << shift_ms
                      << "ms  图像速度≈" << speed << "px/s  等效模糊≈"
                      << speed * blur_ms * 1e-3 << "px\n";
        }
    }
    std::cout<<"valid="<<fit_ok<<" rmse="<<r.reprojection_rmse_px<<"px yaw="<<r.yaw<<" omega="<<r.omega
             <<" center="<<r.center[0]<<","<<r.center[1]<<","<<r.center[2]<<"\n";
    return 0;
}
