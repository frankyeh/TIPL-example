foreach(prog load_nii linear_reg)
  add_executable(${prog} "${prog}.cpp")
  target_link_libraries(${prog} PUBLIC TIPL::tipl)
  set_target_properties(${prog} PROPERTIES CXX_STANDARD 14)
endforeach()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/TIPL-example)
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/TIPL-example/data SYMBOLIC)
