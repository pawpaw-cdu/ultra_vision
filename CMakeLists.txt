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
set(_auto_aim_sim_default OFF)
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
# 放在**根**作用域：io/camera/HikCamera.cpp 要编进 auto_aim，tools/hik_probe 也要用
# 同一个库；原来只在 src/auto_aim 子目录里找，工具目录拿不到 HIK_LIB。
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

add_executable(hsv_tuner tools/hsv_tuner.cpp)

# 下位机链路探针（真机第一步：线/协议/CRC 对不对）。仿真构建也能编，不需要相机。
add_executable(gimbal_probe
    tools/gimbal_probe.cpp
    io/gimbal/gimbal.cpp
    io/gimbal/protocol.cpp
    io/serial/crc16.cpp
    io/serial/serial_port.cpp
)
target_include_directories(gimbal_probe PRIVATE
    ${CMAKE_SOURCE_DIR}/src/auto_aim
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)
target_link_libraries(gimbal_probe PRIVATE ${OpenCV_LIBS} ${YAML_CPP_LDFLAGS} Threads::Threads)

# 海康相机探针：用我们自己的 HikCamera 驱动打开相机、抓帧、存图（需要 MVS SDK）。
if(ULTRA_VISION_USE_HIK_CAMERA)
  add_executable(hik_probe
      tools/hik_probe.cpp
      io/camera/HikCamera.cpp
  )
  target_include_directories(hik_probe PRIVATE
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/io
      ${HIK_INC_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(hik_probe PRIVATE ${OpenCV_LIBS} ${HIK_LIB})
  set_target_properties(hik_probe PROPERTIES BUILD_RPATH "${HIK_LIB_DIR}")
endif()

# ---- 标定工具（对齐 sp_vision 的 calibration/）--------------------------------
# capture：采标定数据（相机画面 + 当下位机云台 yaw/pitch 成对存盘）
# calibrate_camera：棋盘格 → 相机内参
# hand_eye_calibrate：静止标定物 + 多组云台姿态 → 相机→云台外参（手眼）
# （原 calibrate_capture"拍照存盘"工具已删除：现场标定直接在 hand_eye_calibrate --live 里完成）

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

# 配置自查：把 configs/*.yaml 真正加载进结构体的每个值打印出来。
# 改"配置文件 ↔ 代码默认值"时用它做 before/after diff（有效的值必须一行不差），
# 否则很容易出现"yaml 删了一项、hpp 默认却是另一个数"这种跑起来看不出的偏差。
add_executable(config_dump tools/config_dump.cpp)
target_include_directories(config_dump PRIVATE
    ${CMAKE_SOURCE_DIR}/src/auto_aim
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)
target_link_libraries(config_dump PRIVATE ${OpenCV_LIBS} ${YAML_CPP_LDFLAGS})
if(ULTRA_VISION_USE_OPENVINO)
  target_compile_definitions(config_dump PRIVATE ULTRA_VISION_USE_OPENVINO)
  target_link_libraries(config_dump PRIVATE openvino::runtime)
endif()

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
