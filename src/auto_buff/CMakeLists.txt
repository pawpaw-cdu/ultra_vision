# 能量机关（auto_buff）任务：算法 + 薄入口。结构对齐 sp_vision 的 tasks/auto_buff/
#   * auto_buff        —— OBJECT 算法库（感知/解算/估计/瞄准/火控判据 + 诊断），可被
#                         任意入口或测试复用（sp_vision 也是这么拆的）；
#   * auto_buff_demo   —— 薄入口：参数/环境 → 配置 → buff_runner（仿真器或视频）。
# 算法本身不依赖具体入口，也不依赖 OpenVINO 之外的东西。

set(AUTO_BUFF_SOURCES
    rune_types.cpp
    rune_config.cpp
    rune_model.cpp
    rune_refiner.cpp
    rune_plate_refiner.cpp
    rune_blade_state.cpp
    rune_diff_probe.cpp
    rune_diag.cpp
    rune_input.cpp
    rune_slot_lattice.cpp
    rune_round_guard.cpp
    rune_aim_fallback.cpp
    rune_target_selector.cpp
    rune_orbit_fit.cpp
    rune_blade_finder.cpp
    rune_aim_bridge.cpp
    rune_runner.cpp
    rune_detector.cpp
    rune_solver.cpp
    rune_target.cpp
    rune_aimer.cpp
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

# 薄入口（与 auto_aim 的 node_sim.cpp 同构）：只做 参数/环境 → 配置 → runRune。
add_executable(auto_buff_demo buff_demo.cpp $<TARGET_OBJECTS:auto_buff>)
target_link_libraries(auto_buff_demo PRIVATE auto_buff)
