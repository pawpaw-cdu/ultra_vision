# Buff Detector - 能量机关检测与预测系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-14-blue.svg)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.4-green.svg)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey.svg)](https://ubuntu.com/)

> **一个基于几何约束、形态学特征和数据关联的 RoboMaster 能量机关（小符）实时检测与预测系统**

---

## 📖 目录

- [项目简介](#项目简介)
- [环境依赖](#环境依赖)
- [项目结构](#项目结构)
- [算法流程](#算法流程)
- [核心功能](#核心功能)
- [系统架构](#系统架构)

---

## 项目简介

`buff_detector` 是一个专为 **RoboMaster 比赛中小能量机关（小符）识别与击打预测** 设计的 C++ 视觉模块。它利用赛前 3 分钟小符角速度恒定为 60°/s 的先验知识，通过实时检测 **扇叶中心** 和 **“R”标（中心圆）**，结合 **几何约束**、**形态学筛选** 与 **匈牙利算法数据关联**，鲁棒地拟合出能量机关的圆心、半径，并预测所有扇叶的实时位置。

该项目脱胎于 `rm_ultra` 自瞄系统，解决了传统视觉方案中因遮挡、光照变化、扇叶轮廓不稳定导致的 **ID 错乱**、**模型跳变** 和 **丢失后无法快速恢复** 等核心痛点，实现了稳定、低延迟的识别与跟踪。

---

## 环境依赖

- **操作系统: Ubuntu 22.04 (亦可运行于其他 Linux 发行版)**

- **编译器: GCC 9+ (需支持 C++14)**

- **构建工具: CMake 3.10+**

- 核心库:

 **OpenCV 4.x (推荐 4.5.4)**

 **yaml-cpp (用于解析 YAML 配置文件)**

---

## 效果展示

[![点击观看演示视频](screenshots/demo_thumbnail.jpg)](result.mp4)

---

## 项目结构

```Plain
.
 buff_detector/
 ├── CMakeLists.txt          # 构建脚本
 ├── config/                 # 配置文件目录
 │   ├── detector.yaml       # 检测器参数
 │   └── tracker.yaml        # 跟踪器参数
 ├── include/                # 公共头文件
 │   └── energy_detector/
 │       ├── types.hpp       # 数据结构定义
 │       ├── config.hpp      # 配置加载接口
 │       ├── detector.hpp    # 检测器接口
 │       ├── tracker.hpp     # 跟踪器接口
 │       └── utils.hpp       # 可视化工具接口
 ├── src/                    # 源代码
 │   ├── config.cpp          # 配置加载实现
 │   ├── detector.cpp        # 检测器实现
 │   ├── tracker.cpp         # 跟踪器实现
 │   ├── utils.cpp           # 可视化实现
 │   └── main.cpp            # 主程序入口
 └── README.md               # 本文件

```

---

## 算法流程

```Plain
.
 输入图像 → Detector::process()
     ├─ preprocessMask() 生成二值化掩膜
     ├─ extractBladeCenters() 提取扇叶中心
     └─ detectRCenter() 检测“R”标
         ↓
 DetectorResult (扇叶中心 + R标)
         ↓
 Tracker::update()
     ├─ 计算初始圆心/半径
     ├─ 第一帧特殊处理：直接初始化所有扇叶角度
     ├─ 匈牙利匹配：关联当前帧扇叶与跟踪器
     ├─ 角度平滑（指数移动平均）
     ├─ 丢失扇叶的运动预测（恒定角速度）
     └─ 生成 TrackerResult（模型圆心、半径、5个预测扇叶）
         ↓
 可视化：drawModel() + showBladesROI()

```

---

## 核心功能

- **扇叶与“R”标实时检测**
- 基于 BGR 通道差的二值化与形态学处理。
  - 通过轮廓面积、矩形度、圆形度、凸度等多重几何约束，精准筛选扇叶和中心圆。

- **鲁棒的数据关联**
  - 使用 **匈牙利算法**（贪心简化版）将当前帧检测到的扇叶与上一帧的跟踪器进行最优匹配。
  - 有效避免因扇叶临时遮挡或光照突变导致的 ID 错乱和跳变。

- **圆心与半径平滑拟合**
  - 直接利用检测到的“R”标作为圆心候选。
  - 基于所有有效扇叶中心到圆心的平均距离计算半径，并通过指数移动平均（EMA）平滑，消除抖动。

- **扇叶状态预测与丢失恢复**
  - 当某个扇叶丢失时，基于恒定角速度模型外推其角度和位置。
  - 连续丢失超过阈值（`max_lost_frames`）后重置跟踪器，等待重新捕获。

- **YAML 配置驱动**
- 所有算法参数（BGR 通道差阈值、面积范围、平滑系数、匹配阈值等）均通过 `detector.yaml` 和 `tracker.yaml` 配置文件管理，调参无需重新编译。

---

## 系统架构

项目采用清晰的 **分层解耦设计**，主要分为以下三层：

```mermaid

graph TD
    A[主程序 main.cpp] --> B[配置管理 ConfigManager]
    A --> C[检测器 Detector]
    A --> D[跟踪器 Tracker]
    C --> E[预处理: BGR 通道差 + 形态学]
    C --> F[扇叶中心提取]
    C --> G["R"标检测]
    D --> H[匈牙利匹配]
    D --> I[角度平滑与预测]
    D --> J[模型生成 TrackerResult]
    E --> K[掩膜图像]
    F & G --> L[DetectorResult]
    L --> D
    J --> M[可视化 drawModel / showBladesROI]
