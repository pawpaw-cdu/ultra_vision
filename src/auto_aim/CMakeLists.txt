# 仿真入口（auto_aim_sim）始终需要 sim_receiver；真机入口不需要。
if(NOT TARGET sim_receiver)
  add_subdirectory(sim_receiver)
endif()

# ---- 分层：共享能力模块 + 两个薄入口（对应 sp_vision 的 tasks + src/*.cpp）----
# 真机入口与仿真入口**只差 IO**（相机/串口 vs TCP 帧源），能力模块完全共用。
# 所以公共源编成静态库 auto_aim_core，两个入口各建一个可执行文件：
#   auto_aim_real —— 海康/大恒相机 + 串口
#   auto_aim_sim  —— TCP 仿真帧源
# 现场选择启动哪一个即可（两个都在同一份能力库之上，行为一致）。
set(AUTO_AIM_CORE_SOURCES
    control/gimbal_aimer.cpp
    control/gimbal_controller.cpp
    control/aim_signal_filter.cpp
    control/shooter.cpp
    control/target_selector.cpp
    estimation/rotation_rate_estimator.cpp
    estimation/pose_2d_solver.cpp
    perception/detector.cpp
    perception/pnp_solver.cpp
    perception/armor_source.cpp
    control/aim_pipeline.cpp
    Kalman/tracker.cpp
    Kalman/tracker_association.cpp
    Kalman/tracker_watchdog.cpp
    Kalman/tracker_lifecycle.cpp
    Kalman/tracker_accessors.cpp
    Kalman/tracker_scale.cpp
    Kalman/tracker_localfit.cpp
)
set(AUTO_AIM_REAL_ENTRY
    node.cpp
    app/real_runner.cpp
    ${CMAKE_SOURCE_DIR}/io/gimbal/gimbal.cpp
    ${CMAKE_SOURCE_DIR}/io/gimbal/protocol.cpp
    ${CMAKE_SOURCE_DIR}/io/serial/crc16.cpp
    ${CMAKE_SOURCE_DIR}/io/serial/serial_port.cpp
)
set(AUTO_AIM_SIM_ENTRY node_sim.cpp app/sim_runner.cpp)

add_library(auto_aim_core STATIC ${AUTO_AIM_CORE_SOURCES})

# 公共编译选项 / 依赖：对两个入口都生效，挂在 core 上用 PUBLIC 传递。
target_include_directories(auto_aim_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools
    ${CMAKE_SOURCE_DIR}/io
    ${DAHENG_INC_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)
target_compile_definitions(auto_aim_core PUBLIC
    ULTRA_VISION_CONFIG_DIR="${ULTRA_VISION_CONFIG_DIR}"
)
target_link_libraries(auto_aim_core PUBLIC
    ${OpenCV_LIBS}
    ${YAML_CPP_LDFLAGS}
    Threads::Threads
)
if(ULTRA_VISION_USE_OPENVINO)
  target_sources(auto_aim_core PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/nn_detector.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/keypoint_refiner.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/async_detector.cpp
  )
  target_compile_definitions(auto_aim_core PUBLIC ULTRA_VISION_USE_OPENVINO=1)
  target_link_libraries(auto_aim_core PUBLIC openvino::runtime)
endif()

# 一个入口一个可执行文件：auto_aim_sim（TCP 仿真帧源） / auto_aim_real（相机 + 串口）。
# 两者共用同一份 auto_aim_core，只有入口文件与 IO 依赖不同 —— 现场选启动哪个即可。
add_executable(auto_aim_sim ${AUTO_AIM_SIM_ENTRY})
target_link_libraries(auto_aim_sim PRIVATE auto_aim_core sim_receiver)

# 真机入口只依赖一个可用的相机后端：海康优先，缺失时回退大恒。
if(HIK_LIB AND HIK_INC_DIR)
  set(AUTO_AIM_REAL_CAMERA_SOURCE
      ${CMAKE_CURRENT_SOURCE_DIR}/../../io/camera/HikCamera.cpp)
  set(AUTO_AIM_REAL_CAMERA_INCLUDE ${HIK_INC_DIR})
  set(AUTO_AIM_REAL_CAMERA_LIBRARY ${HIK_LIB})
  set(AUTO_AIM_REAL_CAMERA_LIB_DIR ${HIK_LIB_DIR})
  set(AUTO_AIM_REAL_CAMERA_DEFINITION ULTRA_VISION_USE_HIK_CAMERA=1)
elseif(GX_IAPI_LIBRARY AND EXISTS "${DAHENG_INC_DIR}/GxIAPI.h")
  set(AUTO_AIM_REAL_CAMERA_SOURCE
      ${CMAKE_CURRENT_SOURCE_DIR}/../../io/camera/GalaxyCamera.cpp)
  set(AUTO_AIM_REAL_CAMERA_INCLUDE ${DAHENG_INC_DIR})
  set(AUTO_AIM_REAL_CAMERA_LIBRARY ${GX_IAPI_LIBRARY})
  set(AUTO_AIM_REAL_CAMERA_LIB_DIR ${DAHENG_LIB_DIR})
  set(AUTO_AIM_REAL_CAMERA_DEFINITION "")
else()
  set(AUTO_AIM_REAL_CAMERA_SOURCE "")
endif()

if(AUTO_AIM_REAL_CAMERA_SOURCE)
  add_executable(auto_aim_real
      ${AUTO_AIM_REAL_ENTRY}
      ${AUTO_AIM_REAL_CAMERA_SOURCE})
  target_link_libraries(auto_aim_real PRIVATE
      auto_aim_core
      ${AUTO_AIM_REAL_CAMERA_LIBRARY})
  target_include_directories(auto_aim_real PRIVATE ${AUTO_AIM_REAL_CAMERA_INCLUDE})
  if(AUTO_AIM_REAL_CAMERA_DEFINITION)
    target_compile_definitions(auto_aim_real PRIVATE ${AUTO_AIM_REAL_CAMERA_DEFINITION})
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(auto_aim_real PROPERTIES
        BUILD_RPATH "${AUTO_AIM_REAL_CAMERA_LIB_DIR}")
  endif()
else()
  message(STATUS "auto_aim_real 跳过：未找到可用的海康或大恒相机 SDK")
endif()

# ---- 局部拟合器（运行组件；只要找到 Ceres 就编译，与是否构建测试无关）----------
# 定位：EKF 弱点的补强（正面 Yaw/Omega + 旋转中心），异步低频、可注入、默认不接管主链路。
# 与 spin_geometry 一样单独成库、不进 auto_aim_core；Tracker 通过 setLocalFitter 注入。
# 注意：这是**运行组件**，不能藏在 ULTRA_VISION_BUILD_TESTS 里——真机构建 tests=OFF，
# 那样 auto_aim_real 根本编不出拟合器（2026-10-10 发现）。
find_package(Ceres QUIET)
find_package(Eigen3 QUIET)
if(Ceres_FOUND AND Eigen3_FOUND)
  add_library(auto_aim_local_fit STATIC estimation/local_trajectory_fitter.cpp)
  target_include_directories(auto_aim_local_fit PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
      ${Eigen3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(auto_aim_local_fit PUBLIC Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS})
  message(STATUS "auto_aim_local_fit: 启用（Ceres ${Ceres_VERSION}）")
  # 仿真/真机入口都接入（ULTRA_VISION_LOCAL_FIT=1 打开，默认关）。
  target_link_libraries(auto_aim_sim PRIVATE auto_aim_local_fit)
  target_compile_definitions(auto_aim_sim PRIVATE ULTRA_VISION_HAVE_LOCAL_FIT=1)
  if(TARGET auto_aim_real)
    target_link_libraries(auto_aim_real PRIVATE auto_aim_local_fit)
    target_compile_definitions(auto_aim_real PRIVATE ULTRA_VISION_HAVE_LOCAL_FIT=1)
  endif()
else()
  message(STATUS "auto_aim_local_fit: 跳过（未找到 Ceres/Eigen）")
endif()

if(ULTRA_VISION_BUILD_TESTS)
  # ---- 小陀螺灯条轨迹几何估计器（可选组件：仅当找到 Ceres 时编译）----------
  # 定位：异步架构里的"低频全局几何先验/初始化器"，默认**不接管主链路**，
  # 因此单独成库、不进 auto_aim_core（不改变 auto_aim_sim/real 的依赖）。
  find_package(Ceres QUIET)
  if(Ceres_FOUND)
    find_package(Eigen3 REQUIRED)
    add_library(auto_aim_spin_geometry STATIC estimation/spin_geometry_estimator.cpp)
    target_include_directories(auto_aim_spin_geometry PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Eigen3_INCLUDE_DIRS}
    )
    target_link_libraries(auto_aim_spin_geometry PUBLIC Eigen3::Eigen Ceres::ceres)
    message(STATUS "auto_aim_spin_geometry: 启用（Ceres ${Ceres_VERSION}）")

    add_executable(spin_geometry_estimator_test
        test/spin_geometry_estimator_test.cpp
    )
    target_link_libraries(spin_geometry_estimator_test PRIVATE auto_aim_spin_geometry)
    add_test(NAME spin_geometry_estimator_test COMMAND spin_geometry_estimator_test)


    # 局部拟合器的离线工具与单测（库本身已在上面按"是否找到 Ceres"创建）。
    if(TARGET auto_aim_local_fit)
      # 因子图离线原型（第 1 步：每帧独立板位姿）。见 docs/fgo_probe.md。
      add_executable(fgo_probe ${CMAKE_SOURCE_DIR}/tools/fgo_probe.cpp)
      target_include_directories(fgo_probe PRIVATE ${YAML_CPP_INCLUDE_DIRS})
      target_link_libraries(fgo_probe PRIVATE Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS}
          ${YAML_CPP_LDFLAGS})
      add_executable(local_trajectory_fitter_test
          test/local_trajectory_fitter_test.cpp)
      target_include_directories(local_trajectory_fitter_test PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
      target_link_libraries(local_trajectory_fitter_test PRIVATE auto_aim_local_fit)
      add_test(NAME local_trajectory_fitter_test COMMAND local_trajectory_fitter_test)
    endif()
  else()
    message(STATUS "auto_aim_spin_geometry: 跳过（未找到 Ceres；见 docs/spin_geometry_estimator.md）")
  endif()

  add_executable(standard_clock_test
      test/standard_clock_test.cpp
  )
  target_include_directories(standard_clock_test PRIVATE
      ${CMAKE_SOURCE_DIR}
  )
  add_test(NAME standard_clock_test COMMAND standard_clock_test)

  # 开火判据（决策②）：纯逻辑、无状态机/硬件依赖，见 control/shoot_evaluator.hpp
  add_executable(shoot_evaluator_test
      test/shoot_evaluator_test.cpp
  )
  target_include_directories(shoot_evaluator_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME shoot_evaluator_test COMMAND shoot_evaluator_test)


  # 下发通道几何（期望姿态 → C 板命令通道）：钉住"Δ=0 必须等于当前反馈"这条性质，
  # 见 src/auto_aim/control/gimbal_command_frame.hpp 与 docs/auto_aim_aim_loop.md §11。
  add_executable(gimbal_command_frame_test
      test/gimbal_command_frame_test.cpp
  )
  target_include_directories(gimbal_command_frame_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
  )
  add_test(NAME gimbal_command_frame_test COMMAND gimbal_command_frame_test)

  add_executable(aim_signal_filter_test
      test/aim_signal_filter_test.cpp
      control/aim_signal_filter.cpp
  )
  target_include_directories(aim_signal_filter_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME aim_signal_filter_test COMMAND aim_signal_filter_test)

  add_executable(gimbal_aimer_test
      test/gimbal_aimer_test.cpp
      control/gimbal_aimer.cpp
  )
  target_include_directories(gimbal_aimer_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME gimbal_aimer_test COMMAND gimbal_aimer_test)

  add_executable(gimbal_controller_test
      test/gimbal_controller_test.cpp
      control/gimbal_aimer.cpp
      control/gimbal_controller.cpp
  )
  target_include_directories(gimbal_controller_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
  )
  target_link_libraries(gimbal_controller_test PRIVATE Threads::Threads)
  add_test(NAME gimbal_controller_test COMMAND gimbal_controller_test)

  add_executable(shooter_test
      test/shooter_test.cpp
      control/shooter.cpp
  )
  target_include_directories(shooter_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME shooter_test COMMAND shooter_test)

  add_executable(target_selector_test
      test/target_selector_test.cpp
      control/target_selector.cpp
  )
  target_include_directories(target_selector_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME target_selector_test COMMAND target_selector_test)

  add_executable(rotation_rate_estimator_test
      test/rotation_rate_estimator_test.cpp
      estimation/rotation_rate_estimator.cpp
  )
  target_include_directories(rotation_rate_estimator_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME rotation_rate_estimator_test COMMAND rotation_rate_estimator_test)

  add_executable(armor_ekf_test
      test/armor_ekf_test.cpp
      estimation/rotation_rate_estimator.cpp
      # tracker.cpp 的 refineArmorPose 用到 getArmor3DPoints（pnp_solver.cpp），
      # 不列进来会链接失败 —— 之前一直没暴露只是因为那个测试二进制是老的。
      perception/pnp_solver.cpp
      Kalman/tracker.cpp
      Kalman/tracker_association.cpp
      Kalman/tracker_watchdog.cpp
      Kalman/tracker_lifecycle.cpp
      Kalman/tracker_accessors.cpp
      Kalman/tracker_scale.cpp
      Kalman/tracker_localfit.cpp
  )
  target_include_directories(armor_ekf_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(armor_ekf_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME armor_ekf_test COMMAND armor_ekf_test)

  # PnP 降自由度（fit_single）自检：注入角点噪声，比较 IPPE 与 1D yaw 精修的
  # 径向角误差。径向角是 EKF 估 ω / 关联板号的关键观测量。
  add_executable(pnp_yaw_refine_test
      test/pnp_yaw_refine_test.cpp
      perception/pnp_solver.cpp
  )
  target_include_directories(pnp_yaw_refine_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(pnp_yaw_refine_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME pnp_yaw_refine_test COMMAND pnp_yaw_refine_test)

  # 模型构造（整车轮廓重投影）的几何自检：装甲板刚体姿态 + 15° 安装倾角。
  # 把"模型构造 == 局部拟合器的车辆系几何"钉成恒等式，并回归掉历史三个坑
  # （世界系 yaw 造切向 / 观测板 rvec 当竖直 / 丢 tangent.y）。纯头文件，无额外依赖。
  add_executable(armor_geometry_test
      test/armor_geometry_test.cpp
  )
  target_include_directories(armor_geometry_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(armor_geometry_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME armor_geometry_test COMMAND armor_geometry_test)

  # 相位观测自检（审计项 D 的回归）：updateAngle 对四个板位的增益/符号 +
  # updateWithLocalFit 的 yaw / ω 两条通道必须各自独立可用。
  # 纯头文件（armor_ekf.hpp 是实现头），无额外依赖。
  add_executable(armor_ekf_angle_test
      test/armor_ekf_angle_test.cpp
  )
  target_include_directories(armor_ekf_angle_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(armor_ekf_angle_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME armor_ekf_angle_test COMMAND armor_ekf_angle_test)

  # 2D 位姿求解器（6D→4D→2D 的最后一环）自检：拟合器刚体解 → 角点 → 解回来。
  # 只需 pose_2d_solver.cpp + pnp_solver.cpp（要 getArmor3DPoints 的物点定义），
  # 不依赖 Ceres/YAML/相机。
  add_executable(pose_2d_solver_test
      test/pose_2d_solver_test.cpp
      estimation/pose_2d_solver.cpp
      perception/pnp_solver.cpp
  )
  target_include_directories(pose_2d_solver_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(pose_2d_solver_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME pose_2d_solver_test COMMAND pose_2d_solver_test)

  # 局部拟合 yaw 质量门（实测分层：rmse<3px→σ_ψ≈0.6°，8~16px→22.6°，而 ω 各档都稳）。
  # 纯头文件（estimation/local_fit_gate.hpp），无额外依赖。
  # UV-EKF（刚体分解）自检：差分灵敏度 / 更新收敛 / 错板号被 NIS 拒。
  add_executable(uv_ekf_test
      test/uv_ekf_test.cpp
  )
  target_include_directories(uv_ekf_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(uv_ekf_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME uv_ekf_test COMMAND uv_ekf_test)

  add_executable(local_fit_gate_test
      test/local_fit_gate_test.cpp
  )
  target_include_directories(local_fit_gate_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  add_test(NAME local_fit_gate_test COMMAND local_fit_gate_test)

  # 下位机协议层自检：CRC16 标准值 / 43B+29B 字节布局 / 打包解析往返 / 非法帧。
  # 真机"连上但不动"最常见的三个原因（CRC 口径、结构体 padding、帧长）都在这里钉死。
  add_executable(gimbal_protocol_test
      test/gimbal_protocol_test.cpp
      ${CMAKE_SOURCE_DIR}/io/gimbal/protocol.cpp
      ${CMAKE_SOURCE_DIR}/io/serial/crc16.cpp
  )
  target_include_directories(gimbal_protocol_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
  )
  add_test(NAME gimbal_protocol_test COMMAND gimbal_protocol_test)

  # 整链路测试：伪终端冒充下位机，真跑 io::Gimbal（读线程/成帧/CRC 隔离/mode 映射/下发）。
  add_executable(gimbal_link_test
      test/gimbal_link_test.cpp
      ${CMAKE_SOURCE_DIR}/io/gimbal/gimbal.cpp
      ${CMAKE_SOURCE_DIR}/io/gimbal/protocol.cpp
      ${CMAKE_SOURCE_DIR}/io/serial/crc16.cpp
      ${CMAKE_SOURCE_DIR}/io/serial/serial_port.cpp
  )
  target_include_directories(gimbal_link_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
  )
  target_link_libraries(gimbal_link_test PRIVATE Threads::Threads)
  add_test(NAME gimbal_link_test COMMAND gimbal_link_test)

  # Golden-frame detector regression test: pins the output of the classical and
  # the neural detector on recorded frames.
  add_executable(detector_golden_test
      test/detector_golden_test.cpp
      perception/detector.cpp
  )
  target_include_directories(detector_golden_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/common
      ${OpenCV_INCLUDE_DIRS}
      ${YAML_CPP_INCLUDE_DIRS}
  )
  target_compile_definitions(detector_golden_test PRIVATE
      ULTRA_VISION_TEST_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/test/assets"
      ULTRA_VISION_CONFIG_DIR="${ULTRA_VISION_CONFIG_DIR}"
  )
  target_link_libraries(detector_golden_test PRIVATE
      ${OpenCV_LIBS}
      ${YAML_CPP_LDFLAGS}
  )
  if(ULTRA_VISION_USE_OPENVINO)
    target_sources(detector_golden_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/perception/nn_detector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/perception/keypoint_refiner.cpp
    )
    target_compile_definitions(detector_golden_test PRIVATE
        ULTRA_VISION_USE_OPENVINO=1
    )
    target_link_libraries(detector_golden_test PRIVATE openvino::runtime)
  endif()
  add_test(NAME detector_golden_test COMMAND detector_golden_test)

  # sim_receiver 流水线：合成高速发送端，验证"收包 → 解码 → 消费"解耦后客户端
  # 不会被自己的解码拖住、也不会把发送端顶回去（旧实现 recv+imdecode 串在一线程）。
  if(ULTRA_VISION_AUTO_AIM_SIMULATOR AND TARGET sim_receiver)
    add_executable(sim_receiver_pipeline_test
        test/sim_receiver_pipeline_test.cpp
    )
    target_include_directories(sim_receiver_pipeline_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_receiver/src
        ${YAML_CPP_INCLUDE_DIRS}
    )
    target_link_libraries(sim_receiver_pipeline_test PRIVATE
        sim_receiver
        ${YAML_CPP_LDFLAGS}
        Threads::Threads
    )
    add_test(NAME sim_receiver_pipeline_test COMMAND sim_receiver_pipeline_test)
  endif()
endif()
