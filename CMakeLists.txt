cmake_minimum_required(VERSION 3.16)

project(ultra_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ULTRA_VISION_BUILD_AUTO_AIM "Build the auto-aim executable" ON)
option(ULTRA_VISION_BUILD_AUTO_BUFF "Build the energy-buff (auto_buff) task" ON)
option(ULTRA_VISION_BUILD_RUNE "Build the energy-buff (rune) pipeline" ON)
option(ULTRA_VISION_BUILD_TESTS "Build unit tests" OFF)

if(ULTRA_VISION_BUILD_TESTS)
  enable_testing()
endif()

# Use TCP simulator input by default on macOS, where the bundled Galaxy SDK is unavailable.
# On Linux the hardware camera path remains the default.
set(_auto_aim_sim_default ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(_auto_aim_sim_default ON)
endif()
option(ULTRA_VISION_AUTO_AIM_SIMULATOR
       "Use the TCP simulator receiver instead of the Galaxy camera"
       ${_auto_aim_sim_default})
unset(_auto_aim_sim_default)

set(ULTRA_VISION_DAHENG_SDK_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/io/GalaxySDK"
    CACHE PATH "Path to the Daheng Galaxy SDK")
set(ULTRA_VISION_CONFIG_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/configs"
    CACHE PATH "Runtime config directory for auto_aim")

# Homebrew publishes the current major version as `opencv` and 4.x as the
# keg-only `opencv@4`. Prefer opencv@4 when it is present: it is built against
# the ffmpeg the machine currently has, whereas an older `opencv` keg can still
# reference an ffmpeg soname that is no longer installed and then fail to load
# at all (this bit the macOS dev setup: every OpenCV-linked binary, including
# the tests, refused to start with a missing libavformat.*.dylib).
if(NOT DEFINED OpenCV_DIR AND EXISTS "/opt/homebrew/opt/opencv@4/lib/cmake/opencv4")
  set(OpenCV_DIR "/opt/homebrew/opt/opencv@4/lib/cmake/opencv4" CACHE PATH
      "OpenCV CMake config directory")
  message(STATUS "Using the keg-only Homebrew opencv@4")
endif()

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
if(NOT YAML_CPP_LIBRARIES)
  set(YAML_CPP_LIBRARIES ${YAML_CPP_LDFLAGS})
endif()
find_package(Threads REQUIRED)

# Neural-network armor detector. It is optional so that a machine without
# OpenVINO can still build the classical pipeline; the detector itself is
# selected at run time from detector.yaml.
option(ULTRA_VISION_USE_OPENVINO
       "Build the OpenVINO neural-network armor detector" ON)
if(ULTRA_VISION_USE_OPENVINO)
  find_package(OpenVINO QUIET)
  if(NOT OpenVINO_FOUND)
    message(STATUS "OpenVINO not found: neural-network detector disabled")
    set(ULTRA_VISION_USE_OPENVINO OFF)
  else()
    message(STATUS "OpenVINO ${OpenVINO_VERSION}: neural-network detector enabled")
  endif()
endif()

# Energy-rune pipeline (大小能量机关). It reuses the keypoint model already
# exported by sp_vision_25, so it needs OpenVINO, and Eigen for its estimator.
# When either is missing the target is skipped instead of failing the build.
if(ULTRA_VISION_BUILD_RUNE)
  if(NOT ULTRA_VISION_USE_OPENVINO)
    message(STATUS "OpenVINO not found: energy-rune pipeline disabled")
    set(ULTRA_VISION_BUILD_RUNE OFF)
  else()
    find_package(Eigen3 QUIET)
    if(NOT Eigen3_FOUND)
      message(STATUS "Eigen3 not found: energy-rune pipeline disabled")
      set(ULTRA_VISION_BUILD_RUNE OFF)
    else()
      message(STATUS "Eigen3 ${Eigen3_VERSION}: energy-rune pipeline enabled")
    endif()
  endif()
endif()

if(ULTRA_VISION_BUILD_AUTO_AIM AND NOT ULTRA_VISION_AUTO_AIM_SIMULATOR)
  if(NOT EXISTS "${ULTRA_VISION_DAHENG_SDK_ROOT}/include/GxIAPI.h")
    message(FATAL_ERROR
      "Galaxy SDK not found at ${ULTRA_VISION_DAHENG_SDK_ROOT}/include. "
      "Set -DULTRA_VISION_DAHENG_SDK_ROOT=... or enable simulator mode with "
      "-DULTRA_VISION_AUTO_AIM_SIMULATOR=ON.")
  endif()

  set(DAHENG_INC_DIR "${ULTRA_VISION_DAHENG_SDK_ROOT}/include")

  set(ULTRA_VISION_DAHENG_ARCH ""
      CACHE STRING "Galaxy SDK library subdirectory: armv8 or amd64")
  if(ULTRA_VISION_DAHENG_ARCH STREQUAL "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
      set(ULTRA_VISION_DAHENG_ARCH "armv8")
    else()
      set(ULTRA_VISION_DAHENG_ARCH "amd64")
    endif()
  endif()

  set(DAHENG_LIB_DIR
      "${ULTRA_VISION_DAHENG_SDK_ROOT}/lib/${ULTRA_VISION_DAHENG_ARCH}")

  set(GX_IAPI_LIBRARY "${DAHENG_LIB_DIR}/libgxiapi.so")
  if(NOT EXISTS "${GX_IAPI_LIBRARY}")
    message(FATAL_ERROR
      "libgxiapi.so not found in ${DAHENG_LIB_DIR}. "
      "Set -DULTRA_VISION_DAHENG_ARCH=armv8 or amd64 to match the target.")
  endif()

  message(STATUS "Galaxy SDK: ${ULTRA_VISION_DAHENG_SDK_ROOT} (${ULTRA_VISION_DAHENG_ARCH})")
endif()

# ---- 能量机关（auto_buff）------------------------------------------------
# 与自瞄（auto_aim）平行的一个任务：算法在 src/auto_buff/，入口是它自己的
# buff_demo（薄入口）。流水线走 TCP/视频，所以接收层必须先存在。
if(ULTRA_VISION_BUILD_AUTO_BUFF OR ULTRA_VISION_BUILD_RUNE)
  if(NOT TARGET sim_receiver)
    add_subdirectory(src/auto_aim/sim_receiver)
  endif()
  add_subdirectory(src/auto_buff)
endif()

add_executable(hsv_tuner tools/hsv_tuner.cpp)

# 模型延迟/输入尺寸选型探针（原 src/auto_buff/rune_model_bench.cpp，按 sp_vision
# 的做法把工具统一放到 tools/）。
add_executable(buff_model_bench tools/buff_model_bench.cpp)
target_include_directories(buff_model_bench PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(buff_model_bench PRIVATE ${OpenCV_LIBS} openvino::runtime)

# 扇叶状态分类的离线标定探针：编译**同一份** rune_blade_state.cpp，只把
# hub/轨道半径/靶心换成人工标注（网络检不到的"已激活"外观只能人标），
# 用来把"未点亮 / 未激活 / 已激活"的阈值量出来（见 §46）。
add_executable(buff_state_probe
    tools/buff_state_probe.cpp
    src/auto_buff/rune_blade_state.cpp)
target_include_directories(buff_state_probe PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(buff_state_probe PRIVATE ${OpenCV_LIBS})

target_include_directories(hsv_tuner PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(hsv_tuner PRIVATE
    ${OpenCV_LIBS}
)

# Standalone driver for the neural-network detector: run it on an image or a
# video to check the model against real footage or captured simulator frames.
if(ULTRA_VISION_USE_OPENVINO)
  add_executable(nn_probe
      tools/nn_probe.cpp
      src/auto_aim/perception/nn_detector.cpp
      src/auto_aim/perception/keypoint_refiner.cpp
  )
  target_include_directories(nn_probe PRIVATE
      ${CMAKE_SOURCE_DIR}/src/auto_aim
      ${CMAKE_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(nn_probe PRIVATE
      ${OpenCV_LIBS}
      openvino::runtime
  )
endif()

if(ULTRA_VISION_BUILD_AUTO_AIM)
  if(ULTRA_VISION_AUTO_AIM_SIMULATOR)
    message(STATUS "auto_aim input: TCP simulator")
  else()
    message(STATUS "auto_aim input: Galaxy camera")
  endif()
  add_subdirectory(src/auto_aim)
endif()

if(ULTRA_VISION_BUILD_TESTS)
  # 单元测试全部放在顶层 tests/（对齐 sp_vision 的组织方式），算法目录里不放测试。
  add_subdirectory(tests)

  # The accuracy report runs on recorded CSVs; its selftest needs no recording
  # and keeps the metric code itself honest.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME truth_regression_selftest
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_SOURCE_DIR}/tools/truth_regression.py --selftest)
  endif()
endif()
