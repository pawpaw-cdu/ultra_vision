cmake_minimum_required(VERSION 3.15)
project(rm_local_fitter LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV QUIET)
add_executable(rm_demo
  src/main.cpp src/simulator.cpp src/trajectory_fitter.cpp
  src/slow_calib.cpp src/state_solver.cpp src/visualizer.cpp
  src/experiments_multi.cpp)
target_include_directories(rm_demo PRIVATE include)
target_link_libraries(rm_demo PRIVATE Eigen3::Eigen Ceres::ceres)
if(OpenCV_FOUND)
  target_compile_definitions(rm_demo PRIVATE RM_HAVE_OPENCV=1)
  target_include_directories(rm_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(rm_demo PRIVATE ${OpenCV_LIBS})
endif()

# 系统收益量化（纯解析 + 2 状态 Kalman，不依赖 OpenCV/Ceres）
add_executable(value_analysis src/value_analysis.cpp)
target_include_directories(value_analysis PRIVATE include)
target_link_libraries(value_analysis PRIVATE Eigen3::Eigen)
