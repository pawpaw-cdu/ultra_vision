cmake_minimum_required(VERSION 3.16)

project(ultra_vision VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ULTRA_VISION_BUILD_DEBUG "Build the debug instrumentation entry" ON)
option(ULTRA_VISION_BUILD_TESTS "Build unit tests" ON)

set(ULTRA_VISION_CONFIG_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/configs"
    CACHE PATH "Runtime configuration directory")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
if(NOT YAML_CPP_LIBRARIES)
  set(YAML_CPP_LIBRARIES ${YAML_CPP_LDFLAGS})
endif()
find_package(Threads REQUIRED)

add_library(ultra_vision_interfaces INTERFACE)
target_include_directories(ultra_vision_interfaces INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/io/include
    ${CMAKE_CURRENT_SOURCE_DIR}/perception/include
    ${CMAKE_CURRENT_SOURCE_DIR}/estimation/include
    ${CMAKE_CURRENT_SOURCE_DIR}/selection/include
    ${CMAKE_CURRENT_SOURCE_DIR}/control/include
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/energy_rune/include
)

add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(perception)
add_subdirectory(estimation)
add_subdirectory(selection)
add_subdirectory(control)
add_subdirectory(modules/energy_rune)
add_subdirectory(apps/auto_aim)

if(ULTRA_VISION_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
