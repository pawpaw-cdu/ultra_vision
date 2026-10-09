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
    ${CMAKE_CURRENT_SOURCE_DIR}/../../io/camera/GalaxyCamera.cpp
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
add_executable(auto_aim_sim ${AUTO_AIM_CORE_SOURCES} ${AUTO_AIM_SIM_ENTRY})
target_link_libraries(auto_aim_sim PRIVATE auto_aim_core sim_receiver)

# 真机入口需要大恒 Galaxy SDK（相机驱动）；没装就跳过，只出仿真可执行文件。
if(GX_IAPI_LIBRARY AND EXISTS "${DAHENG_INC_DIR}/GxIAPI.h")
  add_executable(auto_aim_real ${AUTO_AIM_CORE_SOURCES} ${AUTO_AIM_REAL_ENTRY})
  target_link_libraries(auto_aim_real PRIVATE auto_aim_core ${GX_IAPI_LIBRARY})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(auto_aim_real PROPERTIES BUILD_RPATH "${DAHENG_LIB_DIR}")
  endif()
else()
  message(STATUS "auto_aim_real 跳过：未找到大恒 Galaxy SDK（GxIAPI.h）")
endif()

# ---- 可选的每目标配置（公共部分已挂在 auto_aim_core 上）--------------------
set(AUTO_AIM_REAL_TARGET auto_aim_real)

  # ---- 可选：海康相机（MVS SDK）--------------------------------------------
  # SDK 的**探测**在根 CMakeLists（Hik SDK 属于 io 层依赖，tools/hik_probe 也要用），
  # 这里只负责把它编进真机入口。
  if(ULTRA_VISION_USE_HIK_CAMERA AND HIK_LIB)
    target_sources(${AUTO_AIM_REAL_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../io/camera/HikCamera.cpp)
    target_include_directories(${AUTO_AIM_REAL_TARGET} PRIVATE ${HIK_INC_DIR})
    target_link_libraries(${AUTO_AIM_REAL_TARGET} PRIVATE ${HIK_LIB})
    target_compile_definitions(${AUTO_AIM_REAL_TARGET} PRIVATE ULTRA_VISION_USE_HIK_CAMERA=1)
    set_target_properties(${AUTO_AIM_REAL_TARGET} PROPERTIES BUILD_RPATH "${HIK_LIB_DIR}")
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

    add_executable(spin_geom_check ${CMAKE_SOURCE_DIR}/tools/spin_geom_check.cpp)
    target_link_libraries(spin_geom_check PRIVATE auto_aim_spin_geometry)

    add_executable(local_fit_from_csv ${CMAKE_SOURCE_DIR}/tools/local_fit_from_csv.cpp)
    target_link_libraries(local_fit_from_csv PRIVATE auto_aim_local_fit)

    # ---- 局部拟合器（小陀螺 θ/ω + 慢层 α/R；可选组件，仅当找到 Ceres 时编译）----
    # 与 spin_geometry 同款：单独成库、不进 auto_aim_core；Tracker 通过注入回调
    # 使用它（见 Kalman/tracker.hpp 的 setLocalFitter）。默认不接管主链路。
    add_library(auto_aim_local_fit STATIC estimation/local_trajectory_fitter.cpp)
    target_include_directories(auto_aim_local_fit PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${Eigen3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(auto_aim_local_fit PUBLIC Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS})
    message(STATUS "auto_aim_local_fit: 启用（Ceres ${Ceres_VERSION}）")

    add_executable(local_trajectory_fitter_test
        test/local_trajectory_fitter_test.cpp
    )
    target_include_directories(local_trajectory_fitter_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
    )
    target_link_libraries(local_trajectory_fitter_test PRIVATE auto_aim_local_fit)
    add_test(NAME local_trajectory_fitter_test COMMAND local_trajectory_fitter_test)

    # 仿真入口接入（P2 A/B）：把拟合器链进 auto_aim_sim，用 ULTRA_VISION_LOCAL_FIT 开关。
    target_link_libraries(auto_aim_sim PRIVATE auto_aim_local_fit)
    target_compile_definitions(auto_aim_sim PRIVATE ULTRA_VISION_HAVE_LOCAL_FIT=1)
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
