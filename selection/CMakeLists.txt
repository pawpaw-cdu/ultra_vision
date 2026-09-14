add_library(ultra_vision_selection STATIC
    src/target_selector.cpp
)

target_include_directories(ultra_vision_selection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ultra_vision_selection PUBLIC
    ultra_vision_interfaces
)
