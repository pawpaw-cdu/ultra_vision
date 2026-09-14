add_library(ultra_vision_app STATIC
    auto_aim_app.cpp
)

target_include_directories(ultra_vision_app PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${YAML_CPP_INCLUDE_DIRS}
)

target_compile_definitions(ultra_vision_app PRIVATE
    ULTRA_VISION_CONFIG_DIR="${ULTRA_VISION_CONFIG_DIR}"
)

target_link_libraries(ultra_vision_app PUBLIC
    ultra_vision_core
    ultra_vision_io
    ultra_vision_perception
    ultra_vision_estimation
    ultra_vision_selection
    ultra_vision_control
    ultra_vision_energy_rune
    ${OpenCV_LIBS}
    ${YAML_CPP_LDFLAGS}
)

add_executable(ultra_vision_auto_aim main.cpp)
target_link_libraries(ultra_vision_auto_aim PRIVATE ultra_vision_app)

if(ULTRA_VISION_BUILD_DEBUG)
  add_executable(ultra_vision_debug ../debug/main.cpp)
  target_link_libraries(ultra_vision_debug PRIVATE ultra_vision_app)
endif()
