add_library(ultra_vision_estimation STATIC
    src/rotation_rate_estimator.cpp
    src/tracker.cpp
)

target_include_directories(ultra_vision_estimation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ultra_vision_estimation PUBLIC
    ultra_vision_core
    ${OpenCV_LIBS}
)
