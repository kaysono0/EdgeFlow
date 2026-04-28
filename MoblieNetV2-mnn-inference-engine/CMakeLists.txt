cmake_minimum_required(VERSION 3.16)
project(learnai_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

set(MNN_ROOT "$ENV{HOME}/workspace/MNN" CACHE PATH "Path to the local MNN source tree")
set(MNN_BUILD_DIR "${MNN_ROOT}/build" CACHE PATH "Path to the local MNN build directory")

find_library(MNN_LIBRARY
    NAMES MNN
    PATHS ${MNN_BUILD_DIR}
    NO_DEFAULT_PATH
)

if(NOT MNN_LIBRARY)
    message(FATAL_ERROR "libMNN was not found. Set -DMNN_BUILD_DIR=/path/to/MNN/build")
endif()

add_executable(infer_utils_test
    tests/infer_utils_test.cpp
)

target_include_directories(infer_utils_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../week03-inference-core/include
)

add_test(NAME infer_utils_test COMMAND infer_utils_test)

add_executable(mnn_mobilenetv2
    src/benchmark.cpp
    src/image_loader.cpp
    src/main.cpp
    src/mnn_classifier.cpp
)

target_include_directories(mnn_mobilenetv2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../week03-inference-core/include
    ${MNN_ROOT}/include
)

target_link_libraries(mnn_mobilenetv2 PRIVATE
    ${MNN_LIBRARY}
    "-framework CoreFoundation"
    "-framework CoreGraphics"
    "-framework ImageIO"
)

set_target_properties(mnn_mobilenetv2 PROPERTIES
    BUILD_RPATH "${MNN_BUILD_DIR}"
)
