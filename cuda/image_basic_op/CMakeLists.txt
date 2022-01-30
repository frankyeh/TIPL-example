cmake_minimum_required(VERSION 3.19)
project(tipl_image_basic_op VERSION "2022.01.23" LANGUAGES CUDA CXX;C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

if(EXISTS "${TIPL_DIR}")
    include_directories(${TIPL_DIR})
else(EXISTS "${TIPL_DIR}")
    find_package(TIPL REQUIRED)
endif(EXISTS "${TIPL_DIR}")

set(SOURCES
    basic_op.cu)

add_executable(tipl_image_basic_op ${SOURCES})

set_target_properties(tipl_image_basic_op PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")

target_compile_definitions(tipl_image_basic_op PUBLIC TIPL_USE_CUDA)
target_link_libraries(tipl_image_basic_op ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

