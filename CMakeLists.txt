cmake_minimum_required(VERSION 3.16)

project(ultra_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ULTRA_VISION_BUILD_AUTO_AIM "Build the auto-aim executable" ON)
option(ULTRA_VISION_BUILD_AUTO_BUFF "Build the energy-buff executable" ON)

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

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
if(NOT YAML_CPP_LIBRARIES)
  set(YAML_CPP_LIBRARIES ${YAML_CPP_LDFLAGS})
endif()
find_package(Threads REQUIRED)

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

if(ULTRA_VISION_BUILD_AUTO_BUFF)
  add_subdirectory(src/auto_buff)
endif()

add_executable(hsv_tuner tools/hsv_tuner.cpp)

target_include_directories(hsv_tuner PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(hsv_tuner PRIVATE
    ${OpenCV_LIBS}
)

if(ULTRA_VISION_BUILD_AUTO_AIM)
  if(ULTRA_VISION_AUTO_AIM_SIMULATOR)
    message(STATUS "auto_aim input: TCP simulator")
  else()
    message(STATUS "auto_aim input: Galaxy camera")
  endif()
  add_subdirectory(src/auto_aim)
endif()
