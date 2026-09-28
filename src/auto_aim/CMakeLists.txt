if(ULTRA_VISION_AUTO_AIM_SIMULATOR)
  if(NOT TARGET sim_receiver)
    add_subdirectory(sim_receiver)
  endif()
  set(AUTO_AIM_SOURCES
      node_sim.cpp
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
  )
else()
  set(AUTO_AIM_SOURCES
      node.cpp
      # 真机链路：控制层（和仿真入口共用同一套类）+ 下位机串口协议。
      control/gimbal_aimer.cpp
      control/gimbal_controller.cpp
      control/aim_signal_filter.cpp
      control/shooter.cpp
      control/target_selector.cpp
      ${CMAKE_SOURCE_DIR}/io/gimbal/gimbal.cpp
      ${CMAKE_SOURCE_DIR}/io/gimbal/protocol.cpp
      ${CMAKE_SOURCE_DIR}/io/serial/crc16.cpp
      ${CMAKE_SOURCE_DIR}/io/serial/serial_port.cpp
      perception/detector.cpp
      perception/pnp_solver.cpp
      perception/armor_source.cpp
      control/aim_pipeline.cpp
      estimation/rotation_rate_estimator.cpp
      Kalman/tracker.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../../io/camera/GalaxyCamera.cpp
  )
endif()

add_executable(auto_aim ${AUTO_AIM_SOURCES})

  # ---- 可选：海康相机（MVS SDK）--------------------------------------------
  # SDK 的**探测**在根 CMakeLists（Hik SDK 属于 io 层依赖，tools/hik_probe 也要用），
  # 这里只负责把它编进 auto_aim。
  if(ULTRA_VISION_USE_HIK_CAMERA AND HIK_LIB)
    target_sources(auto_aim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../io/camera/HikCamera.cpp)
    target_include_directories(auto_aim PRIVATE ${HIK_INC_DIR})
    target_link_libraries(auto_aim PRIVATE ${HIK_LIB})
    target_compile_definitions(auto_aim PRIVATE ULTRA_VISION_USE_HIK_CAMERA=1)
    set_target_properties(auto_aim PROPERTIES BUILD_RPATH "${HIK_LIB_DIR}")
  endif()

if(ULTRA_VISION_USE_OPENVINO)
  target_sources(auto_aim PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/nn_detector.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/keypoint_refiner.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/async_detector.cpp
  )
  target_compile_definitions(auto_aim PRIVATE ULTRA_VISION_USE_OPENVINO=1)
endif()

target_include_directories(auto_aim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools
    ${CMAKE_SOURCE_DIR}/io
    ${DAHENG_INC_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

target_compile_definitions(auto_aim PRIVATE
    ULTRA_VISION_CONFIG_DIR="${ULTRA_VISION_CONFIG_DIR}"
)

target_link_libraries(auto_aim PRIVATE
    ${OpenCV_LIBS}
    ${YAML_CPP_LDFLAGS}
    Threads::Threads
)

if(ULTRA_VISION_AUTO_AIM_SIMULATOR)
  target_link_libraries(auto_aim PRIVATE sim_receiver)
else()
  target_link_libraries(auto_aim PRIVATE ${GX_IAPI_LIBRARY})
endif()

if(ULTRA_VISION_USE_OPENVINO)
  target_link_libraries(auto_aim PRIVATE openvino::runtime)
endif()

if(NOT ULTRA_VISION_AUTO_AIM_SIMULATOR AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(auto_aim PROPERTIES BUILD_RPATH "${DAHENG_LIB_DIR}")
endif()

if(ULTRA_VISION_BUILD_TESTS)
  add_executable(standard_clock_test
      test/standard_clock_test.cpp
  )
  target_include_directories(standard_clock_test PRIVATE
      ${CMAKE_SOURCE_DIR}
  )
  add_test(NAME standard_clock_test COMMAND standard_clock_test)

  # UV（像素重投影）观测的自检：纯数学，钉住重投影几何与"喂真值必须收敛"。
  # 见 src/auto_aim/test/armor_ekf_uv_test.cpp 与 docs/uv_observation.md。
  # 开火判据（决策②）：纯逻辑、无状态机/硬件依赖，见 control/shoot_evaluator.hpp
  add_executable(shoot_evaluator_test
      test/shoot_evaluator_test.cpp
  )
  target_include_directories(shoot_evaluator_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME shoot_evaluator_test COMMAND shoot_evaluator_test)

  add_executable(armor_ekf_uv_test
      test/armor_ekf_uv_test.cpp
  )
  target_include_directories(armor_ekf_uv_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME armor_ekf_uv_test COMMAND armor_ekf_uv_test)

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
  )
  target_include_directories(armor_ekf_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(armor_ekf_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME armor_ekf_test COMMAND armor_ekf_test)

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
endif()
