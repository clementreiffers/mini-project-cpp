cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
project(cours4)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH}
        "${CMAKE_SOURCE_DIR}/dependencies/opencv/build/")


find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})