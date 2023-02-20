cmake_minimum_required(VERSION 3.23.0)
project(tipl_image_resample VERSION "2022.01.23" LANGUAGES CUDA CXX;C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(SOURCES
    resample.cu)

add_executable(tipl_image_resample ${SOURCES})

set_target_properties(tipl_image_resample PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)
set_property(TARGET tipl_image_resample PROPERTY CUDA_ARCHITECTURES all)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

target_compile_definitions(tipl_image_resample PUBLIC TIPL_USE_CUDA)
if(TIPL_HOME)
    target_include_directories(tipl_basic_op PUBLIC ${TIPL_HOME})
else ()
    target_link_libraries(tipl_basic_op TIPL::tipl)
endif()

target_link_libraries(tipl_image_resample ${CUDA_LIBRARIES})

