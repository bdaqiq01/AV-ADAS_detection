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
    src/lane/LaneDetect.cpp
)

target_include_directories(adas PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lane
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(adas PRIVATE ${OpenCV_LIBS})
