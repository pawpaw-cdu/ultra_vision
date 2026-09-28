# 深大 RP-26Rune 能量机关算法的仿真 harness（独立工程，不参与 Ultra_Vision 主构建）。
#
#   cmake -S tools/rp26_sim -B build_rp26_sim \
#         -DRP26_ROOT=/Users/a/Desktop/RM_source/RP-26Rune-main \
#         -DCMAKE_MODULE_PATH=/tmp/cmake_modules
#   cmake --build build_rp26_sim -j 4
#
# 编译进本工程的源文件全部来自 RP-26Rune-main（原样编译，不改他们的仓库）：
#   * src/core/algorithm/power_rune  （算法本体 + interface）
#   * src/core/algorithm/function    （时间戳工具）
#   * src/core/utility/transform_tools（TFTree / Angle）
#   * src/core/utility/config_loader （ReJson：读他们的 JSON 配置）
# 可视化与插件框架用 tools/rp26_sim/include 下的影子头文件替换（见各文件注释）。

cmake_minimum_required(VERSION 3.22.0)
project(rp26_sim CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(RP26_ROOT "/Users/a/Desktop/RM_source/RP-26Rune-main" CACHE PATH "RP-26Rune-main 仓库根目录")
if(NOT EXISTS "${RP26_ROOT}/src/core/algorithm/power_rune")
    message(FATAL_ERROR "RP26_ROOT 不正确：${RP26_ROOT}")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
find_package(OpenVINO REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(Sophus REQUIRED)

set(RP26_POWER_RUNE_ROOT "${RP26_ROOT}/src/core/algorithm/power_rune")
set(RP26_INCLUDE_DIRS
    "${RP26_POWER_RUNE_ROOT}/include"
    "${RP26_POWER_RUNE_ROOT}/interface"
    "${RP26_ROOT}/src/core/algorithm/function/inc"
    "${RP26_ROOT}/src/core/utility/transform_tools/inc"
    "${RP26_ROOT}/src/core/utility/time/inc"
    "${RP26_ROOT}/src/core/utility/config_loader/inc"
)

file(GLOB_RECURSE RP26_POWER_RUNE_SRC
    "${RP26_POWER_RUNE_ROOT}/src/*.cpp"
)
# RuneObservationRefiner.cpp 用 harness 里的"只加日志"副本（rp26_refiner_probe.cpp）
# 替换，避免同一份符号被编译两次。
list(FILTER RP26_POWER_RUNE_SRC EXCLUDE REGEX "RuneObservationRefiner\\.cpp$")
list(APPEND RP26_POWER_RUNE_SRC "${RP26_POWER_RUNE_ROOT}/interface/power_rune_interface.cpp")

file(GLOB_RECURSE RP26_FUNCTION_SRC "${RP26_ROOT}/src/core/algorithm/function/src/*.cpp")
file(GLOB_RECURSE RP26_TRANSFORM_SRC "${RP26_ROOT}/src/core/utility/transform_tools/src/*.cpp")
file(GLOB_RECURSE RP26_CONFIG_LOADER_SRC
    "${RP26_ROOT}/src/core/utility/config_loader/src/*.cpp")

add_executable(rp26_sim
    src/main.cpp
    src/rp26_detector.cpp
    src/rp26_refiner_probe.cpp
    src/sim_link.cpp
    ${RP26_POWER_RUNE_SRC}
    ${RP26_FUNCTION_SRC}
    ${RP26_TRANSFORM_SRC}
    ${RP26_CONFIG_LOADER_SRC}
)

# 离线几何对照工具：在同一批录制帧上比"他们的轮廓几何"和"我们的关键点几何"。
add_executable(rp26_geom_compare
    src/geom_compare.cpp
    src/rp26_detector.cpp
    src/rp26_refiner_probe.cpp
    ${RP26_FUNCTION_SRC}
    ${RP26_TRANSFORM_SRC}
    ${RP26_CONFIG_LOADER_SRC}
)

# 影子头文件必须排在最前：json.hpp / PnPVariable.hpp / img_viz.hpp /
# foxglove_viz/foxglove_viz.hpp / Time_generated.h 都要命中这里的版本。
set(RP26_HARNESS_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${RP26_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

set(RP26_HARNESS_LIBS
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
    glog::glog
    openvino::runtime
    Sophus::Sophus
)

foreach(target rp26_sim rp26_geom_compare)
    target_include_directories(${target} PRIVATE ${RP26_HARNESS_INCLUDES})
    target_link_libraries(${target} PRIVATE ${RP26_HARNESS_LIBS})
    target_compile_options(${target} PRIVATE -fPIC)
    target_compile_definitions(${target} PRIVATE
        RP26_DEFAULT_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config"
        RP26_HARNESS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endforeach()

set_target_properties(rp26_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(rp26_geom_compare PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 配置同步做成独立 target，避免"可执行文件没重新链接、POST_BUILD 不触发"导致
# 改了 config/ 却没生效（踩过这个坑：跑出来的是旧参数）。
add_custom_target(rp26_sim_config ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/config"
            "${CMAKE_BINARY_DIR}/config"
    COMMENT "sync rp26_sim config"
)
add_dependencies(rp26_sim rp26_sim_config)
add_dependencies(rp26_geom_compare rp26_sim_config)
