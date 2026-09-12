if(ULTRA_VISION_AUTO_AIM_SIMULATOR)
  add_subdirectory(sim_receiver)
  set(AUTO_AIM_SOURCES
      node_sim.cpp
      perception/detector.cpp
      perception/pnp_solver.cpp
      Kalman/tracker.cpp
  )
else()
  set(AUTO_AIM_SOURCES
      node.cpp
      perception/detector.cpp
      perception/pnp_solver.cpp
      Kalman/tracker.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../../io/camera/GalaxyCamera.cpp
  )
endif()

add_executable(auto_aim ${AUTO_AIM_SOURCES})

target_include_directories(auto_aim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools
    ${CMAKE_SOURCE_DIR}/io
    ${DAHENG_INC_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

target_compile_definitions(auto_aim PRIVATE
    ULTRA_VISION_CONFIG_DIR="${ULTRA_VISION_CONFIG_DIR}"
)

target_link_libraries(auto_aim PRIVATE
    ${OpenCV_LIBS}
    ${YAML_CPP_LDFLAGS}
    Threads::Threads
)

if(ULTRA_VISION_AUTO_AIM_SIMULATOR)
  target_link_libraries(auto_aim PRIVATE sim_receiver)
else()
  target_link_libraries(auto_aim PRIVATE ${GX_IAPI_LIBRARY})
endif()

if(NOT ULTRA_VISION_AUTO_AIM_SIMULATOR AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(auto_aim PROPERTIES BUILD_RPATH "${DAHENG_LIB_DIR}")
endif()
