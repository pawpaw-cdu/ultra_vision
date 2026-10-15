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

# ---- 海康（Hikrobot MVS）相机 SDK 探测 ---------------------------------------
# 放在**根**作用域：io/camera/HikCamera.cpp 要编进 auto_aim，根作用域才能把它
# 的 include/lib 交给子目录（原来只在 src/auto_aim 里找，别处拿不到 HIK_LIB）。
option(ULTRA_VISION_USE_HIK_CAMERA "Build the Hikvision (MVS SDK) camera driver" OFF)
set(ULTRA_VISION_HIK_SDK_ROOT "/opt/MVS" CACHE PATH "MVS SDK root (include/ + lib/)")
set(HIK_LIB "")
set(HIK_LIB_DIR "")
if(ULTRA_VISION_USE_HIK_CAMERA)
  find_path(HIK_INC_DIR MvCameraControl.h HINTS "${ULTRA_VISION_HIK_SDK_ROOT}/include")
  # MVS 的 lib/ 下同时有 32/64（x86）或 32/64/aarch64，**必须先挑匹配本机字长的**：
  # 原来 `lib/*/...` 的 glob 在 x86_64 上会先抓到 lib/32/...，链接出的二进制一跑就
  # "wrong ELF class: ELFCLASS32"（或加载失败）。按架构显式排序。
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      set(HIK_ARCH_DIRS "aarch64" "64")
    else()
      set(HIK_ARCH_DIRS "64" "x86_64")
    endif()
  else()
    set(HIK_ARCH_DIRS "32")
  endif()
  foreach(HIK_ARCH ${HIK_ARCH_DIRS})
    foreach(HIK_SUFFIX "" ".1" ".2")
      if(NOT HIK_LIB AND EXISTS
         "${ULTRA_VISION_HIK_SDK_ROOT}/lib/${HIK_ARCH}/libMvCameraControl.so${HIK_SUFFIX}")
        set(HIK_LIB
            "${ULTRA_VISION_HIK_SDK_ROOT}/lib/${HIK_ARCH}/libMvCameraControl.so${HIK_SUFFIX}")
      endif()
    endforeach()
  endforeach()
  if(HIK_INC_DIR AND HIK_LIB)
    get_filename_component(HIK_LIB_DIR "${HIK_LIB}" DIRECTORY)
    message(STATUS "Hik camera: ${HIK_INC_DIR} + ${HIK_LIB}")
  else()
    message(WARNING "ULTRA_VISION_USE_HIK_CAMERA=ON 但没找到 MVS SDK（${ULTRA_VISION_HIK_SDK_ROOT}，架构目录候选 ${HIK_ARCH_DIRS}），跳过")
    set(HIK_LIB "")
  endif()
endif()

# 可执行文件统一放到**构建目录根部**（build/auto_aim 而不是 build/src/auto_aim/auto_aim）。
# 理由：多个任务（auto_aim / auto_buff / 各种探针）混在一棵树里时，深层相对路径又长又容易
# 在脚本/文档里写错；平铺到 build/ 后"跑哪个、在哪"一眼可见。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

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

# ---- 标定工具（对齐 sp_vision_25 的 calibration/）-----------------------------
# sp_vision_25 有 calibrate_camera + calibrate_handeye(+robotworld) 这一套，我们的
# 两个工具是同一职责（内参 / 相机→云台手眼），所以保留；与它不重合的一次性探针
# 已经在 uv 收敛时删掉。
# calibrate_camera：棋盘格/圆点阵 → 相机内参（有海康 SDK 时还能 --live 直接拍）
# hand_eye_calibrate：静止标定物 + 多组云台姿态 → 相机→云台外参（手眼）

add_executable(calibrate_camera tools/calibrate_camera.cpp)
target_include_directories(calibrate_camera PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/io
    ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
target_link_libraries(calibrate_camera PRIVATE ${OpenCV_LIBS} ${YAML_CPP_LDFLAGS})
# 有海康 SDK 时把 --live 采集（直接开相机拍标定板）编进去
if(ULTRA_VISION_USE_HIK_CAMERA AND HIK_LIB)
  target_sources(calibrate_camera PRIVATE io/camera/HikCamera.cpp)
  target_include_directories(calibrate_camera PRIVATE ${HIK_INC_DIR})
  target_compile_definitions(calibrate_camera PRIVATE ULTRA_VISION_USE_HIK_CAMERA=1)
  target_link_libraries(calibrate_camera PRIVATE ${HIK_LIB})
  set_target_properties(calibrate_camera PROPERTIES BUILD_RPATH "${HIK_LIB_DIR}")
endif()

add_executable(hand_eye_calibrate
    tools/hand_eye_calibrate.cpp
    src/auto_aim/perception/detector.cpp
    src/auto_aim/perception/pnp_solver.cpp
    src/auto_aim/perception/armor_source.cpp
    io/gimbal/gimbal.cpp io/gimbal/protocol.cpp io/serial/crc16.cpp io/serial/serial_port.cpp
)
target_include_directories(hand_eye_calibrate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/auto_aim ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/io
    ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS}
)
target_link_libraries(hand_eye_calibrate PRIVATE ${OpenCV_LIBS} ${YAML_CPP_LDFLAGS} Threads::Threads)
if(ULTRA_VISION_USE_OPENVINO)
  target_sources(hand_eye_calibrate PRIVATE
      src/auto_aim/perception/nn_detector.cpp
      src/auto_aim/perception/keypoint_refiner.cpp
      src/auto_aim/perception/async_detector.cpp)
  target_compile_definitions(hand_eye_calibrate PRIVATE ULTRA_VISION_USE_OPENVINO=1)
  target_link_libraries(hand_eye_calibrate PRIVATE openvino::runtime)
endif()
# 有海康 SDK 时把手眼工具的 --live（直接开相机 + 串口 + 实时解算）也编进去
if(ULTRA_VISION_USE_HIK_CAMERA AND HIK_LIB)
  target_sources(hand_eye_calibrate PRIVATE io/camera/HikCamera.cpp)
  target_include_directories(hand_eye_calibrate PRIVATE ${HIK_INC_DIR})
  target_compile_definitions(hand_eye_calibrate PRIVATE ULTRA_VISION_USE_HIK_CAMERA=1)
  target_link_libraries(hand_eye_calibrate PRIVATE ${HIK_LIB})
  set_target_properties(hand_eye_calibrate PROPERTIES BUILD_RPATH "${HIK_LIB_DIR}")
endif()

# ---- sp_vision_25 的滤波器 / 底盘构造 / 重投影（**移植版**）--------------------
# 为什么搬：用户在真机上验证过 sp_vision_25 的底盘模型构造与重投影"非常优秀"，
# 而我们自己这套的几何学习等补丁在实车上会漂（实测 l 在 18s 内从 +3.5cm 走到
# −16cm）。所以底座换成它的实现，局部拟合器只在这个底座上做 A/B。
# 只搬了 armor/target/tracker/solver + extended_kalman_filter/math_tools；
# 裁掉全向感知重载与 spdlog（换零依赖桩）；滤波器逻辑一行没动，namespace 改 spv。
# 见 src/spv/tools/logger.hpp 与 git 历史里的 docs/spv_port.md。
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  add_library(spv_core STATIC
      src/spv/armor.cpp
      src/spv/target.cpp
      src/spv/tracker.cpp
      src/spv/solver.cpp
      src/spv/tools/extended_kalman_filter.cpp
      src/spv/tools/math_tools.cpp)
  target_include_directories(spv_core PUBLIC
      ${CMAKE_SOURCE_DIR}/src/spv ${CMAKE_SOURCE_DIR}/src
      ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
  target_link_libraries(spv_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS} ${YAML_CPP_LDFLAGS})
  message(STATUS "spv_core: 启用（sp_vision_25 的 tracker/solver 移植版）")
else()
  message(STATUS "spv_core: 跳过（未找到 Eigen3）")
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
endif()
