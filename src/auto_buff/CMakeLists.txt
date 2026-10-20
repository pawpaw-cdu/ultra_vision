# 能量机关任务：按 app/config/common/detection/estimation/control/support 分层。

set(AUTO_BUFF_SOURCES
    common/rune_types.cpp
    config/rune_config.cpp
    detection/rune_model.cpp
    detection/rune_refiner.cpp
    detection/rune_plate_refiner.cpp
    detection/rune_detector.cpp
    estimation/rune_solver.cpp
    estimation/rune_target.cpp
    estimation/rune_slot_lattice.cpp
    control/rune_aimer.cpp
    control/rune_aim_bridge.cpp
    control/rune_aim_fallback.cpp
    control/rune_blade_selector.cpp
    control/rune_fire_gate.cpp
    control/rune_round_guard.cpp
    app/rune_input.cpp
    app/rune_runner.cpp
    # 云台轨迹生成与角度解算复用自瞄那一套（100 Hz、加加速度受限）
    ${CMAKE_SOURCE_DIR}/src/auto_aim/control/gimbal_aimer.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_aim/control/gimbal_controller.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_aim/control/aim_signal_filter.cpp
)

add_library(auto_buff OBJECT ${AUTO_BUFF_SOURCES})

target_include_directories(auto_buff PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/auto_aim
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

target_compile_definitions(auto_buff PUBLIC
    ULTRA_VISION_CONFIG_DIR="${ULTRA_VISION_CONFIG_DIR}"
)

target_link_libraries(auto_buff PUBLIC
    sim_receiver
    ${OpenCV_LIBS}
    ${YAML_CPP_LDFLAGS}
    Eigen3::Eigen
    openvino::runtime
    Threads::Threads
)

# 薄入口：参数/环境 → 配置 → runRune。
add_executable(auto_buff_demo app/buff_demo.cpp $<TARGET_OBJECTS:auto_buff>)
target_link_libraries(auto_buff_demo PRIVATE auto_buff)
