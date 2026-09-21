cmake_minimum_required(VERSION 3.10)
project(sim_receiver LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

add_library(sim_receiver STATIC
    src/tcp_input.cpp
    src/vision_date_recevier.cpp
    src/telemetry_receiver.cpp
)

target_include_directories(sim_receiver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_SOURCE_DIR}
)

target_include_directories(sim_receiver PRIVATE ${YAML_CPP_INCLUDE_DIRS})
target_link_libraries(sim_receiver PUBLIC ${OpenCV_LIBS})

include(GNUInstallDirs)
install(TARGETS sim_receiver
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
