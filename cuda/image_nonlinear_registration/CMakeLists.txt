cmake_minimum_required(VERSION 3.19)
project(tipl_image_nonlinear_registration VERSION "2022.01.23" LANGUAGES CUDA CXX;C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(SOURCES
    nonlinear_registration.cu)

add_executable(tipl_image_nonlinear_registration ${SOURCES})

set_target_properties(tipl_image_nonlinear_registration PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --extended-lambda")

target_compile_definitions(tipl_image_nonlinear_registration PUBLIC TIPL_USE_CUDA)
if( USE_INSTALLED_TIPL )
  target_link_libraries(tipl_image_nonlinear_registration TIPL::tipl)
else ()
  target_include_directories(tipl_image_nonlinear_registration PUBLIC ${TIPL_HOME})
endif()

target_link_libraries(tipl_image_nonlinear_registration ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

