add_library(ultra_vision_io STATIC
    src/tcp_input.cpp
    src/vision_date_recevier.cpp
)

target_include_directories(ultra_vision_io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(ultra_vision_io PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${YAML_CPP_INCLUDE_DIRS}
)

target_link_libraries(ultra_vision_io PUBLIC
    ultra_vision_core
    ${OpenCV_LIBS}
    ${YAML_CPP_LDFLAGS}
)
