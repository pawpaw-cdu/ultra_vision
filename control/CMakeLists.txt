add_library(ultra_vision_control STATIC
    src/aim_signal_filter.cpp
    src/gimbal_aimer.cpp
    src/gimbal_controller.cpp
    src/shooter.cpp
)

target_include_directories(ultra_vision_control PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ultra_vision_control PUBLIC
    ultra_vision_core
    ultra_vision_selection
    Threads::Threads
)
