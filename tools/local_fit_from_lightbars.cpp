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
#include "estimation/local_trajectory_fitter.hpp"
#include <algorithm>
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
            double uu0,vv0,uu1,vv1; proj(v[i0],v[i1],v[i2],uu0,vv0); proj(v[i3],v[i4],v[i5],uu1,vv1);
            // top = 图像上端（投影后 v 更小的端点）；(x0,y0,z0)/(x1,y1,z1) 来自网格顶点序，
            // 实测恒有 y0>y1（相机 y 向下）→ x0 端是下端，直接当 top 会造成 ~L 的系统偏差。
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
    std::cout<<"valid="<<fit_ok<<" rmse="<<r.reprojection_rmse_px<<"px yaw="<<r.yaw<<" omega="<<r.omega
             <<" center="<<r.center[0]<<","<<r.center[1]<<","<<r.center[2]<<"\n";
    return 0;
}
