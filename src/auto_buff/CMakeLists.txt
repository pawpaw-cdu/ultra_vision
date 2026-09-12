set(ENERGY_DEMO_SOURCES
    src/main.cpp
    src/config.cpp
    src/detector.cpp
    src/tracker.cpp
    src/utils.cpp
    src/pnp_solver.cpp
)

add_executable(energy_demo ${ENERGY_DEMO_SOURCES})

target_include_directories(energy_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

target_compile_definitions(energy_demo PRIVATE
    ULTRA_VISION_BUFF_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config"
    ULTRA_VISION_BUFF_VIDEO_PATH="${CMAKE_CURRENT_SOURCE_DIR}/result.mp4"
)

target_link_libraries(energy_demo PRIVATE
    ${OpenCV_LIBS}
    ${YAML_CPP_LDFLAGS}
)
