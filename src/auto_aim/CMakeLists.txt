if(ULTRA_VISION_AUTO_AIM_SIMULATOR)
  if(NOT TARGET sim_receiver)
    add_subdirectory(sim_receiver)
  endif()
  set(AUTO_AIM_SOURCES
      node_sim.cpp
      control/gimbal_aimer.cpp
      control/gimbal_controller.cpp
      control/aim_signal_filter.cpp
      control/shooter.cpp
      control/target_selector.cpp
      estimation/rotation_rate_estimator.cpp
      perception/detector.cpp
      perception/pnp_solver.cpp
      Kalman/tracker.cpp
  )
else()
  set(AUTO_AIM_SOURCES
      node.cpp
      perception/detector.cpp
      perception/pnp_solver.cpp
      estimation/rotation_rate_estimator.cpp
      Kalman/tracker.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../../io/camera/GalaxyCamera.cpp
  )
endif()

add_executable(auto_aim ${AUTO_AIM_SOURCES})

if(ULTRA_VISION_USE_OPENVINO)
  target_sources(auto_aim PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/nn_detector.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/keypoint_refiner.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/perception/async_detector.cpp
  )
  target_compile_definitions(auto_aim PRIVATE ULTRA_VISION_USE_OPENVINO=1)
endif()

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

if(ULTRA_VISION_USE_OPENVINO)
  target_link_libraries(auto_aim PRIVATE openvino::runtime)
endif()

if(NOT ULTRA_VISION_AUTO_AIM_SIMULATOR AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(auto_aim PROPERTIES BUILD_RPATH "${DAHENG_LIB_DIR}")
endif()

if(ULTRA_VISION_BUILD_TESTS)
  add_executable(standard_clock_test
      test/standard_clock_test.cpp
  )
  target_include_directories(standard_clock_test PRIVATE
      ${CMAKE_SOURCE_DIR}
  )
  add_test(NAME standard_clock_test COMMAND standard_clock_test)

  add_executable(aim_signal_filter_test
      test/aim_signal_filter_test.cpp
      control/aim_signal_filter.cpp
  )
  target_include_directories(aim_signal_filter_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME aim_signal_filter_test COMMAND aim_signal_filter_test)

  add_executable(gimbal_aimer_test
      test/gimbal_aimer_test.cpp
      control/gimbal_aimer.cpp
  )
  target_include_directories(gimbal_aimer_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME gimbal_aimer_test COMMAND gimbal_aimer_test)

  add_executable(gimbal_controller_test
      test/gimbal_controller_test.cpp
      control/gimbal_aimer.cpp
      control/gimbal_controller.cpp
  )
  target_include_directories(gimbal_controller_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
  )
  target_link_libraries(gimbal_controller_test PRIVATE Threads::Threads)
  add_test(NAME gimbal_controller_test COMMAND gimbal_controller_test)

  add_executable(shooter_test
      test/shooter_test.cpp
      control/shooter.cpp
  )
  target_include_directories(shooter_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME shooter_test COMMAND shooter_test)

  add_executable(target_selector_test
      test/target_selector_test.cpp
      control/target_selector.cpp
  )
  target_include_directories(target_selector_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME target_selector_test COMMAND target_selector_test)

  add_executable(rotation_rate_estimator_test
      test/rotation_rate_estimator_test.cpp
      estimation/rotation_rate_estimator.cpp
  )
  target_include_directories(rotation_rate_estimator_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME rotation_rate_estimator_test COMMAND rotation_rate_estimator_test)

  add_executable(armor_ekf_test
      test/armor_ekf_test.cpp
      estimation/rotation_rate_estimator.cpp
      Kalman/tracker.cpp
  )
  target_include_directories(armor_ekf_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(armor_ekf_test PRIVATE ${OpenCV_LIBS})
  add_test(NAME armor_ekf_test COMMAND armor_ekf_test)

  # Golden-frame detector regression test: pins the output of the classical and
  # the neural detector on recorded frames.
  add_executable(detector_golden_test
      test/detector_golden_test.cpp
      perception/detector.cpp
  )
  target_include_directories(detector_golden_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/common
      ${OpenCV_INCLUDE_DIRS}
      ${YAML_CPP_INCLUDE_DIRS}
  )
  target_compile_definitions(detector_golden_test PRIVATE
      ULTRA_VISION_TEST_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/test/assets"
      ULTRA_VISION_CONFIG_DIR="${ULTRA_VISION_CONFIG_DIR}"
  )
  target_link_libraries(detector_golden_test PRIVATE
      ${OpenCV_LIBS}
      ${YAML_CPP_LDFLAGS}
  )
  if(ULTRA_VISION_USE_OPENVINO)
    target_sources(detector_golden_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/perception/nn_detector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/perception/keypoint_refiner.cpp
    )
    target_compile_definitions(detector_golden_test PRIVATE
        ULTRA_VISION_USE_OPENVINO=1
    )
    target_link_libraries(detector_golden_test PRIVATE openvino::runtime)
  endif()
  add_test(NAME detector_golden_test COMMAND detector_golden_test)
endif()
