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
    Kalman/tracker_lifecycle.cpp
    Kalman/tracker_accessors.cpp
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

if(ULTRA_VISION_BUILD_TESTS)

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
      Kalman/tracker_lifecycle.cpp
      Kalman/tracker_accessors.cpp
  )
  target_include_directories(armor_ekf_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(armor_ekf_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME armor_ekf_test COMMAND armor_ekf_test)

  # 模型构造（整车轮廓重投影）的几何自检：装甲板刚体姿态 + 15° 安装倾角。
  # 把模型构造的车辆系几何钉成恒等式，并回归掉历史三个坑
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

  # 相位观测自检：updateAngle 对四个板位的增益与符号必须一致。
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
