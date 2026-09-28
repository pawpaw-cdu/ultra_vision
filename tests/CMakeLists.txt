# 单元测试（对齐 sp_vision 的 tests/：测试不进算法目录，统一放这里）。
#
#   buff_target_test  —— 估计器/瞄准的数值行为（解析观测，无图像、无仿真器）
#   buff_state_test   —— 纯状态类：回合/保持/窗口、槽位记账、瞄准兜底、椭圆拟合

add_executable(buff_target_test
    buff_target_test.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_buff/rune_types.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_buff/rune_target.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_buff/rune_aimer.cpp
)
target_include_directories(buff_target_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)
target_link_libraries(buff_target_test PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
add_test(NAME buff_target_test COMMAND buff_target_test)

add_executable(buff_state_test
    buff_state_test.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_buff/rune_aim_fallback.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_buff/rune_orbit_fit.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_buff/rune_round_guard.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_buff/rune_slot_lattice.cpp
    ${CMAKE_SOURCE_DIR}/src/auto_buff/rune_target_selector.cpp
)
target_include_directories(buff_state_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(buff_state_test PRIVATE ${OpenCV_LIBS})
add_test(NAME buff_state_test COMMAND buff_state_test)
