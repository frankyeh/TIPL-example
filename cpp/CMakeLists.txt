foreach(prog load_nii linear_reg)
  add_executable(${prog} "${prog}.cpp")
  if( USE_INSTALLED_TIPL )
  	target_link_libraries(${prog} PUBLIC  TIPL::tipl)
  else()
    target_include_directories(${prog} PUBLIC ${TIPL_HOME})
    if( Threads_FOUND ) 	
	    target_link_libraries( ${prog} PUBLIC Threads::Threads )
    endif()
  endif()
  set_target_properties(${prog} PROPERTIES CXX_STANDARD 17)
endforeach()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/TIPL-example)
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/TIPL-example/data SYMBOLIC)
