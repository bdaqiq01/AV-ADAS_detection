cmake_minimum_required(VERSION 3.16)
project(adas_detection LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)

add_executable(adas
    app/main.cpp
    app/frameProcessor.cpp
    src/lane/LaneDetect.cpp
    src/stop_speedLimit/yoloDetector.cpp
    src/pedestrian/PedestrianDetector.cpp
)

target_include_directories(adas PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lane
    ${CMAKE_CURRENT_SOURCE_DIR}/src/stop_speedLimit
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pedestrian
    ${CMAKE_CURRENT_SOURCE_DIR}/app
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/cuda-12.6/targets/aarch64-linux/include
    /usr/src/tensorrt/include
)

target_link_directories(adas PRIVATE
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda-12.6/targets/aarch64-linux/lib
)

target_link_libraries(adas PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
)

target_link_options(adas PRIVATE
    -Wl,--unresolved-symbols=ignore-in-shared-libs
)
