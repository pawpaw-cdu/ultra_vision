add_library(ultra_vision_perception STATIC
    src/detector.cpp
    src/pnp_solver.cpp
)

target_include_directories(ultra_vision_perception PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ultra_vision_perception PUBLIC
    ultra_vision_core
    ${OpenCV_LIBS}
)
