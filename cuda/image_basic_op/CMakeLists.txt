cmake_minimum_required(VERSION 3.19)
project(tipl_image_basic_op VERSION "2022.01.23" LANGUAGES CUDA CXX;C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(SOURCES
    basic_op.cu)

add_executable(tipl_image_basic_op ${SOURCES})

set_target_properties(tipl_image_basic_op PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)
set_property(TARGET tipl_image_basic_op PROPERTY CUDA_ARCHITECTURES 52 60 70 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

target_compile_definitions(tipl_image_basic_op PUBLIC TIPL_USE_CUDA)
if( USE_INSTALLED_TIPL ) 
	target_link_libraries(tipl_image_basic_op TIPL::tipl)
else ()
  target_include_directories(tipl_image_basic_op PUBLIC ${TIPL_HOME})
endif()

target_link_libraries(tipl_image_basic_op ${CUDA_LIBRARIES})

