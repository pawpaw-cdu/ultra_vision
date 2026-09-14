add_library(ultra_vision_core STATIC
    src/logger.cpp
)

target_include_directories(ultra_vision_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ultra_vision_core PUBLIC
    ultra_vision_interfaces
    Threads::Threads
)
