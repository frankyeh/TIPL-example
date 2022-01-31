cmake_minimum_required(VERSION 3.19)
project(tipl_image_linear_registration VERSION "2022.01.23" LANGUAGES CUDA CXX;C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(SOURCES
    linear_registration.cu)

add_executable(tipl_image_linear_registration ${SOURCES})

set_target_properties(tipl_image_linear_registration PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")

target_compile_definitions(tipl_image_linear_registration PUBLIC TIPL_USE_CUDA)
find_package(TIPL)
if ( NOT TIPL_FOUND )
  if(EXISTS "${TIPL_DIR}")
    target_include_directories(tipl_image_linear_registration ${TIPL_DIR})
  endif()
else()
  target_link_libraries(tipl_image_linear_registration TIPL::tipl)
endif()

target_link_libraries(tipl_image_linear_registration ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
