cmake_minimum_required(VERSION 3.16)

project(week03_ncnn_conversion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT NCNN_DIR)
    set(NCNN_DIR "/Users/junsen/workspace/ncnn/build/install/lib/cmake/ncnn")
endif()

set(ncnn_DIR "${NCNN_DIR}" CACHE PATH "NCNN package directory" FORCE)
find_package(ncnn REQUIRED CONFIG)

add_executable(mobilenetv2_ncnn
    src/mobilenetv2_ncnn.cpp
)

target_include_directories(mobilenetv2_ncnn PRIVATE
    /Users/junsen/workspace/ncnn/build/install/include
    /Users/junsen/workspace/ncnn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../week03-inference-core/include
)

target_link_libraries(mobilenetv2_ncnn PRIVATE ncnn)

if(APPLE)
    target_compile_options(mobilenetv2_ncnn PRIVATE -O3)
endif()
