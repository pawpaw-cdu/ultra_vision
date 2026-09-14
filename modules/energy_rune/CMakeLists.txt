add_library(ultra_vision_energy_rune INTERFACE)
target_link_libraries(ultra_vision_energy_rune INTERFACE
    ultra_vision_interfaces
    ${OpenCV_LIBS}
)
