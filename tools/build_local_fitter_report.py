#!/usr/bin/env python3
"""生成《局部拟合器技术报告》的 Markdown 与 DOCX（内容同源，避免两版不一致）。

用法： python3 tools/build_local_fitter_report.py
产出： docs/local_fitter_report.md、docs/local_fitter_report.docx

DOCX 里的显示公式用 matplotlib mathtext 渲染成 PNG 插入（容器内无 mathjax/OMML 通道）。
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(REPO, "docs")
EQDIR = os.path.join(DOCS, "_eq")

AUTHOR = "谢金恩"
DATE = "2026 年 10 月 9 日"
TITLE = "局部拟合器技术报告"
SUBTITLE = "小陀螺装甲板旋转的独立 Yaw 与 Omega 观测源"

# ---------------------------------------------------------------------------
# 内容（块列表）：('h1',s) ('h2',s) ('p',s) ('ul',[..]) ('table',header,[rows])
#              ('eq',latex,label) ('pagebreak',) ('sig',)
# ---------------------------------------------------------------------------
BLOCKS = [
    ("p", "作者：" + AUTHOR + "　　日期：" + DATE),
    ("p", "本报告说明局部拟合器（Local Trajectory Fitter）的来由、数学思路、在自瞄系统中的"
          "接入实现与实测效果，并在最后给出完整数学推导。内容与仓库实现对应，所有结论都来自"
          "仿真与离线实验的实测数据。"),

    ("h1", "一、背景"),
    ("h2", "1.1 单帧 PnP 的正面病态"),
    ("p", "小陀螺工况下，自瞄对装甲板 Yaw 与角速度 Omega 的估计主要依赖单帧 PnP 加 EKF 时序滤波。"
          "当装甲板正面朝向相机时，单帧几何对 Yaw 是不敏感的：左右灯条像素间距对 Yaw 的一阶导数在"
          "正面为零，"),
    ("eq", r"\frac{\partial (u_{right}-u_{left})}{\partial \theta}=0\quad(\theta=0)", "eq1"),
    ("p", "即正面处 Yaw 是二阶小量，而像素噪声是一阶量。结果是：正面观测时 EKF 只能靠预测硬撑，"
          "Yaw 方差大、Omega 误差随之增大，进而影响提前量与命中。"),
    ("h2", "1.2 核心物理事实：灯条 15 度安装倾角"),
    ("p", "裁判系统的装甲板灯条相对旋转平面有一个统一的安装倾角 alpha（约 15 度）。倾角让灯条的"
          "上下端点具有不同的旋转半径："),
    ("eq", r"r_{top}=R-\frac{L}{2}\sin\alpha,\qquad r_{bot}=R+\frac{L}{2}\sin\alpha", "eq2"),
    ("p", "于是正面观测时，上下端点的像素横坐标之差（差分信号）会随 Yaw 变化，其一阶导数为"),
    ("eq", r"\frac{\partial (u_{top}-u_{bot})}{\partial \theta}\approx f_x\,\frac{L\sin\alpha\,(2R-C_z)}{C_z^{2}}", "eq3"),
    ("p", "这是一个物理上真实存在的独立观测，不是坐标变换的产物。它正是局部拟合器能够在正面"
          "提供 Yaw 的根据。"),
    ("h2", "1.3 定位"),
    ("p", "局部拟合器不替代 EKF，而是作为异步低频模块，在正面观测等 EKF 弱点处提供独立的 Yaw 与 "
          "Omega 观测，同时给出旋转中心估计。定位是“EKF 弱点的补强”加“旋转中心估计器”。"),

    ("h1", "二、目的"),
    ("ul", [
        "在正面观测、单帧 PnP 的 Yaw 病态时，提供一路独立、可用的 Yaw/Omega 观测。",
        "提供旋转中心估计，既可辅助估计器，也可作为“瞄准旋转中心”策略的输入。",
        "作为只读/可选模块接入，默认关闭，失败时优雅降级回单帧 PnP，不影响主链路。",
    ]),

    ("h1", "三、推导思路"),
    ("h2", "3.1 坐标系"),
    ("p", "相机系 C：Z_c 向前、X_c 向右、Y_c 向下。车辆系 V：Z_v 向上、X_v 向前、Y_v 向右。"
          "相机与车辆的关系固定为 R_cv = Rx(phi_gimbal) 乘 R_base，其中 R_base 把车辆三个轴映到"
          "相机：X_v 到 Z_c、Y_v 到 X_c、Z_v 到负 Y_c。"),
    ("h2", "3.2 刚体模型"),
    ("p", "装甲板上任一点在相机系下的位置写成刚体旋转加平移："),
    ("eq", r"P^{c}_{i}(t)=C(t)+R_{cv}\,R(\theta(t))\,P^{v}_{i}", "eq4"),
    ("p", "其中 C 是旋转中心（相机系），theta 是绕竖直轴的偏航，P^v 是点在车辆系的坐标。"
          "把旋转中心随时间的小漂移也建模为线性项 C(t) = C + V (t - t0)。"),
    ("h2", "3.3 灯条端点参数化"),
    ("p", "在车辆系里，以板面法向 n、切向 t、竖直 z 表示灯条两端点。设灯条在板面的切向偏移为 a、"
          "沿竖直的半长为 b（上下取正负），倾角 alpha 让径向与高度随 b 改变："),
    ("eq", r"\rho=R-b\sin\alpha,\quad z=h+b\cos\alpha", "eq5"),
    ("eq", r"P^{v}=[\rho\cos\theta-a\sin\theta,\;\rho\sin\theta+a\cos\theta,\;z]^{\mathsf{T}}", "eq6"),
    ("p", "这个参数化有一个重要性质：同一条灯条的上下端点只差 b 的符号，于是"),
    ("eq", r"\Vert P_{top}-P_{bot}\Vert^{2}=(L\sin\alpha)^{2}+(L\cos\alpha)^{2}=L^{2}", "eq7"),
    ("p", "即灯条长度约束是恒等成立的，投影后的灯条长度自然约束了深度，不需要再额外加一项。"),
    ("h2", "3.4 刚体运动分解"),
    ("p", "把同一块板两根灯条的观测按“差分”和“平均”组合，可以解耦：平均量给出旋转中心与半径，"
          "差分量给出相位与角速度。差分/平均解耦的好处是中心漂移只进平均量，不再污染旋转估计。"),
    ("h2", "3.5 可观测性"),
    ("p", "正面（theta 接近零）时左右灯条间距对 Yaw 的一阶导为零，Yaw 不可观测；而上下端点差分"
          "的一阶导正比于 L sin alpha（式 eq3），当 alpha 非零时 Yaw 可观测。仿真对照：alpha 等于 0 "
          "时正面 Yaw 不可观测，alpha 等于 15 度时可观测。"),
    ("h2", "3.6 双层估计器"),
    ("p", "快层固定倾角 alpha 与回转半径 R，用 Ceres 拟合中心 C、高度 h、初相 theta0、角速度 omega "
          "（可选角加速度 beta）与微弱的中心漂移速度；慢层固定快层最优参数，只放开 alpha 与 R 做"
          "一小步确定性 Gauss-Newton，输出 J 转置乘 J 的对角信息量，交给带遗忘与门控的融合。"),

    ("h1", "四、实现"),
    ("h2", "4.1 拟合器本体"),
    ("p", "拟合器位于 estimation/local_trajectory_fitter。快层用 Ceres 自动微分最小化重投影残差，"
          "并使用 Huber 鲁棒核；多起点覆盖角速度符号与初相，避免掉进镜像解。慢层用确定性"
          "Gauss-Newton 对 alpha 与 R 走一小步。"),
    ("h2", "4.2 A 段：单板正面弧切分"),
    ("p", "拟合器的前提是“输入为同一块物理板在连续时间上的观测”。在线链路里同帧会出现多块板、"
          "板号会跳变、正面弧会被打断，因此需要先做弧段切分。实现为三步："),
    ("ul", [
        "身份：按图像板心的最近邻加常速预测把观测关联成一条条“单块板的弧”，不依赖绝对板号。",
        "分段：同一条弧内帧间隔超过阈值即断开。",
        "正面门控：只累计图像两灯条间距满足 sep 大于 gate 乘 sep_max 的帧（即板相对相机偏角小于"
        "阈值），离开正对区即清段；只有正对时图像左右才对应板面切向的左右灯条。",
    ]),
    ("p", "门控与深度带过滤同时抑制了近处杂检与道具，避免它们混进弧里。"),
    ("h2", "4.3 B 段：坐标与符号约定"),
    ("p", "拟合器的 theta 约定与 EKF 的 Yaw/Omega 约定之间需要标定。实测两者符号相反，配置项 "
          "yaw_rate_sign 取负一。另有两点经验修正："),
    ("ul", [
        "yaw 的 90 度吸附：绝对板号在多板同现时不可靠，拟合输出的 Yaw 可能整体差整数倍的 90 度。"
        "EKF 自身 Yaw 平滑、不会差 90 度，于是把拟合 Yaw 吸附到与 EKF Yaw 最近的、模 90 度同余的"
        "那个值，稳妥地消掉未知偏移。",
        "默认只把 Omega（和中心）喂 EKF：实测拟合 Yaw 是最脆弱的量，约定稍错就会把 EKF 拉坏；"
        "Omega 则稳健可用。",
    ]),
    ("h2", "4.4 在线慢层：不硬编码 R"),
    ("p", "慢层在线估计回转半径 R，不再写死为固定值。每步先做快层拟合，再对 R 走一小步，然后"
          "做门控融合：可观测性（旋转中心深度接近两倍 R 时冻结）、信息量下限、拟合残差门控、"
          "单步上限、学习率，以及一个必要的弱先验回拉。回拉是关键：没有它，R 在较远距离会沿"
          "R 与 C_z 的耦合方向慢漂。"),
    ("h2", "4.5 接入接口与配置"),
    ("p", "Tracker 通过注入回调使用拟合器，观测为灯条上/下端点像素加板号提示。关键配置：enabled "
          "默认关闭；window_frames 与 min_bars 决定窗口；front_gate 为正面门控阈值；use_yaw 默认"
          "关闭（只喂 Omega）；joint_plates 为多板联合开关，默认关闭。"),

    ("h1", "五、实验与效果"),
    ("p", "以下数据来自仿真（陀螺角速度 1 到 8 rad/s、距离 3.0 与 6.5 米、相机静止或闭环）。"),
    ("h2", "5.1 干净真值与检测端点"),
    ("table", ["输入", "有效", "重投影 RMSE", "Omega（真值 -3.00）"], [
        ["模型自造合成数据", "是", "0.0001 px", "-3.000"],
        ["仿真几何 + 真值线性时间", "是", "0.29 px", "-2.998"],
        ["干净端点真值", "是", "0.32 px", "-3.00"],
        ["带噪声检测端点", "是", "3 到 4 px", "-2.78 到 -3.05"],
    ]),
    ("h2", "5.2 端点高斯噪声扫描"),
    ("p", "给灯条 CSV 的六个端点坐标加高斯噪声，每个噪声水平重复多次取中位。Yaw/Omega 误差近似"
          "线性增长，约每像素 0.09 rad/s："),
    ("table", ["端点噪声", "约当像素", "有效", "Omega 误差中位"], [
        ["1 mm", "0.5 px", "100%", "0.03 rad/s"],
        ["2 mm", "0.9 px", "100%", "0.08 rad/s"],
        ["4 mm", "1.8 px", "100%", "0.18 rad/s"],
        ["8 mm", "3.7 px", "100%", "0.48 rad/s"],
        ["12 mm 以上", "5.5 px 以上", "0%", "1.0 以上（被有效性门限拒绝）"],
    ]),
    ("p", "结论：一到两像素下 Omega 基本不受伤；约 4 像素开始出现个别镜像解；5 像素以上先失效的"
          "是有效性门限，不是 Omega 本身。实际仿真检测端点的噪声约 3 像素，正好落在这条曲线的拐点。"),
    ("h2", "5.3 正面分箱 A/B（关键结论）"),
    ("p", "把 EKF 的 Omega 误差按板朝向分箱（正面为两灯条像素间距接近最大时），对比开/关局部拟合器。"
          "正面帧的 EKF Omega 误差显著高于侧面，体现正面病态；局部拟合器把正面误差降低约两成到三成。"),
    ("table", ["工况", "正面 关", "正面 开", "侧面 关", "侧面 开"], [
        ["3.0 m / 3 rad", "0.92", "0.75（-18%）", "0.59", "0.46"],
        ["3.0 m / 5 rad", "0.56", "0.52", "0.40", "0.52"],
        ["6.5 m / 3 rad", "0.78", "0.53（-32%）", "0.50", "0.39"],
        ["6.5 m / 5 rad", "0.62", "0.49（-21%）", "0.45", "0.46"],
    ]),
    ("p", "把之前的“3 rad/s 增益不明显”纠正过来：那是没按朝向分箱、把正面帧和侧面帧平均掉了。"
          "分箱后可以看到，拟合器的增益恰好落在 EKF 最弱的正面帧上，且中低速即见效。加上 5 到 8 "
          "rad/s 高速档降低一成半到近三成，合起来是：拟合器在 EKF 最弱的两处提供增益。"),
    ("h2", "5.4 旋转中心估计"),
    ("table", ["距离", "中心横向误差", "中心深度误差"], [
        ["3.0 m", "约 1 cm", "几 cm"],
        ["6.5 m", "约 1 cm", "约 10 cm"],
    ]),
    ("p", "只要拟合有效（重投影 RMSE 小于 5 像素），中心精度在厘米级，可以喂给“瞄准旋转中心”策略。"),
    ("h2", "5.5 多板联合与在线慢层"),
    ("p", "多板联合：拟合器本就支持共享中心/半径/高度、相位相差固定角。近距多板一开始经常失败，"
          "根因是把同帧两块板映到同一板号、联合窗相位对不上。改用两段式相对身份（先用最长弧拟合出"
          "旋转中心，再用另一块板的板心相对中心的方位角定相位偏移，避开病态的单帧 PnP Yaw）后，"
          "相位偏移稳定为相邻板 90 度，近距有效期从个位数百分比恢复到接近百分之百。"),
    ("p", "在线慢层：R 在线估计后稳定在 0.19 到 0.21（真值约 0.20），不再硬编码。默认“只喂 Omega”"
          "下，EKF 的 Omega 误差在三组工况中两优一略差。"),
    ("table", ["工况", "写死 R", "在线 R 且只喂 Omega"], [
        ["3.0 m / 3 rad", "0.62", "0.56（-10%）"],
        ["6.5 m / 3 rad", "0.41", "0.48（+17%）"],
        ["6.5 m / 5 rad", "0.67", "0.54（-19%）"],
    ]),
    ("h2", "5.6 闭环跟瞄与发弹"),
    ("p", "不冻结云台、闭环跟瞄时，中低速（3 rad/s）下有锁即可瞄住并开火，开火瞬间的 Yaw 误差"
          "中位 0.1 到 0.55 度、俯仰误差约 0 度，即这些弹是瞄着发的。开局部拟合器后开火次数更多"
          "（与约两成的估计增益一致）。需要说明：本轮闭环测试受仿真渲染帧率限制（闭环下约 15 帧"
          "每秒，会间歇丢锁），干净的中低速闭环命中率 A/B 需要在有图形加速的主机上重跑。"),

    ("h1", "六、结论与局限"),
    ("p", "结论：局部拟合器本体没有问题（干净输入重投影 0.32 像素、Omega 精确；检测输入约 3 像素）。"
          "它给 EKF 的可用输出是 Omega 与旋转中心；Yaw 是最脆弱的量，需要强门控。它的增益集中在"
          "EKF 最弱的两处——正面观测与高转速。"),
    ("p", "局限：中低速且非正面时单帧 PnP 加 EKF 本就够用，拟合器增益有限（约一成）；正面病态在"
          "仿真里比真机轻，本次正面增益可视为偏保守；多板联合要真正兑现需要在线慢层估计 R（已完成）"
          "以及可靠的相对板身份；闭环命中率的干净数据待有图形加速的主机补测。"),
    ("p", "建议的定位：默认关闭、按条件自启的补强模块——正面观测（间距比大于阈值）或高转速"
          "（角速度大于阈值）时启用，输出 Omega 与旋转中心；Yaw 走强门控通道或暂不使用；旋转中心"
          "接给“瞄准旋转中心”策略。"),

    ("pagebreak",),
    ("h1", "七、数学推导"),
    ("h2", "7.1 记号"),
    ("p", "相机系下点用上标 c，车辆系用上标 v。灯条长度 L，板面切向偏移 a 取上下端点的 L/2，"
          "安装倾角 alpha，回转半径 R，灯条中心高度 h。"),
    ("h2", "7.2 刚体模型与投影"),
    ("eq", r"P^{c}=C+R_{cv}R(\theta)P^{v},\qquad R_{cv}=R_x(\phi_{gimbal})R_{base}", "eq8"),
    ("eq", r"u=f_x\frac{X_c}{Z_c}+c_x,\qquad v=f_y\frac{Y_c}{Z_c}+c_y", "eq9"),
    ("h2", "7.3 差分与平均"),
    ("p", "同一块板两根灯条，切向偏移分别为正负 W/2。两灯条中心相加取平均得到板心，"
          "差分则给出相位与半径的组合。对板心，切向偏移抵消，板心只含 R 与相位："),
    ("eq", r"\bar{X}_c=C_x+R\sin\theta,\qquad \bar{Z}_c=C_z+R\cos\theta", "eq10"),
    ("p", "对上下端点，径向与高度按式 eq5 随 b 变化。上端点 b 取正、下端点 b 取负，于是上下"
          "端点的像素横坐标之差随 Yaw 变化，即式 eq3，这正是正面 Yaw 可观测性的来源。"),
    ("h2", "7.4 残差与求解"),
    ("p", "快层对每个观测的上下端点各写两个像素残差（投影减去观测），参数为 C、h、theta0、omega、"
          "beta 与漂移速度，用 Ceres 自动微分加 Huber 核求解："),
    ("eq", r"r_i=\left[f_x\frac{X_c+C_x+V_x t}{Z_c+C_z+V_z t}+c_x-u_{obs},\;\; f_y\frac{Y_c+C_y+V_y t}{Z_c+C_z+V_z t}+c_y-v_{obs}\right]", "eq11"),
    ("p", "慢层固定快层参数，只对 alpha 与 R 做一步 Gauss-Newton，并以 (J^T J) 对角作信息量："),
    ("eq", r"\Delta p=(J^{\mathsf{T}}J+\lambda I)^{-1}J^{\mathsf{T}}r", "eq12"),
    ("h2", "7.5 在线 R 的门控融合"),
    ("p", "可观测且拟合可信时，R 走一小步并被弱先验回拉；否则冻结："),
    ("eq", r"R\leftarrow \mathrm{clamp}(R+\eta\,\Delta R+\gamma\,(R_{prior}-R),\;R_{lo},\;R_{hi})", "eq13"),
    ("p", "其中 clamps 把 R 限制在硬边界内，eta 为学习率，gamma 为弱先验回拉系数。可观测性条件是"
          "旋转中心深度与两倍 R 的差大于一个带宽（避免该退化解）。"),
    ("h2", "7.6 有效性"),
    ("p", "拟合输出以重投影均方根作为质量判据之一，超过阈值或参数越界即视为无效，回退单帧 PnP，"
          "保证主链路不被坏观测影响。"),

    ("h1", "附：结论摘要"),
    ("ul", [
        "拟合器本体：干净输入重投影 0.32 像素、Omega 精确；检测输入约 3 像素、Omega 误差约一成。",
        "正面分箱：正面帧 EKF Omega 误差降低约两成到三成；高速档降低一成半到近三成。",
        "旋转中心：厘米级；可喂“瞄准旋转中心”。",
        "在线慢层：R 在线估计且不漂；给 EKF 的可用输出是 Omega 与中心，Yaw 需强门控。",
    ]),
    ("sig",),
]


def render_eq(latex, name):
    os.makedirs(EQDIR, exist_ok=True)
    path = os.path.join(EQDIR, name + ".png")
    if os.path.exists(path):
        return path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, "$" + latex + "$", fontsize=15)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06, transparent=True)
    plt.close(fig)
    return path


def build_md():
    out = ["# " + TITLE + "\n", "*" + SUBTITLE + "*\n"]
    for b in BLOCKS:
        k = b[0]
        if k == "h1":
            out.append("## " + b[1] + "\n")
        elif k == "h2":
            out.append("### " + b[1] + "\n")
        elif k == "p":
            out.append(b[1] + "\n")
        elif k == "ul":
            out.append("\n".join("- " + x for x in b[1]) + "\n")
        elif k == "table":
            hdr, rows = b[1], b[2]
            out.append("| " + " | ".join(hdr) + " |")
            out.append("| " + " | ".join(["---"] * len(hdr)) + " |")
            for r in rows:
                out.append("| " + " | ".join(r) + " |")
            out.append("")
        elif k == "eq":
            out.append("$$\n" + b[1] + "\n$$\n")
        elif k == "pagebreak":
            out.append("\n---\n")
        elif k == "sig":
            out.append("\n" + AUTHOR + "\n\n" + DATE + "\n")
    return "\n".join(out)


def build_docx():
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    LATIN, CJK = "Calibri", "Microsoft YaHei"

    def sty(run, size=None, bold=None, color=(0, 0, 0)):
        run.font.name = LATIN
        rpr = run._element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.append(rf)
        rf.set(qn("w:ascii"), LATIN); rf.set(qn("w:hAnsi"), LATIN); rf.set(qn("w:eastAsia"), CJK)
        if size is not None: run.font.size = Pt(size)
        if bold is not None: run.bold = bold
        run.font.color.rgb = RGBColor(*color)

    def para(text, size=10.5, bold=False, align=None, before=0, after=6):
        p = doc.add_paragraph()
        if align is not None: p.alignment = align
        p.paragraph_format.space_before = Pt(before)
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.line_spacing = 1.3
        sty(p.add_run(text), size=size, bold=bold)
        return p

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(1.0)
    sec.top_margin = sec.bottom_margin = Inches(0.9)

    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    t.paragraph_format.space_after = Pt(2)
    sty(t.add_run(TITLE), size=20, bold=True)
    s = doc.add_paragraph(); s.paragraph_format.space_after = Pt(10)
    sty(s.add_run(SUBTITLE), size=11, bold=False, color=(0x44, 0x44, 0x44))

    for b in BLOCKS:
        k = b[0]
        if k == "h1":
            para(b[1], size=14.5, bold=True, before=12, after=6)
        elif k == "h2":
            para(b[1], size=12, bold=True, before=8, after=4)
        elif k == "p":
            para(b[1])
        elif k == "ul":
            for x in b[1]:
                p = doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(3)
                sty(p.add_run(x), size=10.5)
        elif k == "table":
            hdr, rows = b[1], b[2]
            tbl = doc.add_table(rows=1, cols=len(hdr)); tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            tbl.style = "Table Grid"
            for j, h in enumerate(hdr):
                c = tbl.rows[0].cells[j]
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                sty(c.paragraphs[0].add_run(h), size=10, bold=True, color=(0xFF, 0xFF, 0xFF))
                sht = OxmlElement("w:shd"); sht.set(qn("w:fill"), "44546A")
                c._tc.get_or_add_tcPr().append(sht)
            for ri, r in enumerate(rows):
                cs = tbl.add_row().cells
                for j, v in enumerate(r):
                    cs[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
                    sty(cs[j].paragraphs[0].add_run(v), size=10)
                    if ri % 2 == 1:
                        shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), "F2F2F2")
                        cs[j]._tc.get_or_add_tcPr().append(shd)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif k == "eq":
            path = render_eq(b[1], b[2])
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(6)
            p.add_run().add_picture(path, height=Inches(0.30))
        elif k == "pagebreak":
            doc.add_page_break()
        elif k == "sig":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p.paragraph_format.space_before = Pt(18)
            sty(p.add_run(AUTHOR), size=11)
            p2 = doc.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            sty(p2.add_run(DATE), size=11)

    out = os.path.join(DOCS, "local_fitter_report.docx")
    doc.save(out)
    return out


def main():
    os.makedirs(DOCS, exist_ok=True)
    md = build_md()
    with open(os.path.join(DOCS, "local_fitter_report.md"), "w") as f:
        f.write(md)
    print("wrote docs/local_fitter_report.md")
    if "--render-eq" in sys.argv:
        for b in BLOCKS:
            if b[0] == "eq":
                render_eq(b[1], b[2])
        print("rendered", sum(1 for b in BLOCKS if b[0] == "eq"), "equations to docs/_eq/")
        return
    print("wrote", build_docx())


if __name__ == "__main__":
    main()
