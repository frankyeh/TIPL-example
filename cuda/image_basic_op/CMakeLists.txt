cmake_minimum_required(VERSION 3.19)
project(tipl_basic_op VERSION "2022.01.23" LANGUAGES CUDA CXX;C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(SOURCES
    basic_op.cu)

add_executable(tipl_basic_op ${SOURCES})

set_target_properties(tipl_basic_op PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)
set_property(TARGET tipl_basic_op PROPERTY CUDA_ARCHITECTURES all)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23.0")
    set_property(TARGET tipl_basic_op PROPERTY CUDA_ARCHITECTURES all)
else()
    set_property(TARGET tipl_basic_op PROPERTY CUDA_ARCHITECTURES "50;52;53;60;61;62;70;72;75;80;86")
endif()

target_compile_definitions(tipl_basic_op PUBLIC TIPL_USE_CUDA)
if(TIPL_HOME)
    target_include_directories(tipl_basic_op PUBLIC ${TIPL_HOME})
else ()
    target_link_libraries(tipl_basic_op TIPL::tipl)
endif()

target_link_libraries(tipl_basic_op ${CUDA_LIBRARIES})

