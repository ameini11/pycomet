function(ConfigTarget)

  # Sources
  # ---------------------------------------------------------------------------
  target_sources(${TARGET_NAME}
    PRIVATE
      ${SOURCE_FILES_C}
      ${SOURCE_FILES_F}
  )

  # Include directories
  # ---------------------------------------------------------------------------
  target_include_directories(${TARGET_NAME}
    PRIVATE
      ${tensorflow_INCLUDE_DIR}
      ${cppflow_INCLUDE_DIR}
      ${json_INCLUDE_DIR}
      # where the library itself will look for its internal headers
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
      # where top-level project will look for the library's public headers
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      # where external projects will look for the library's public headers
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  # Debug suffix
  # ---------------------------------------------------------------------------
  # It might be a good idea to add d suffix to debug binaries - that way you’ll
  # get lib<name>.a with Debug configuration and lib<name>.a with Release. To
  # do that you need to set the DEBUG_POSTFIX property.
  set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX "d")

  # External libraries
  # ---------------------------------------------------------------------------
  target_link_libraries(${TARGET_NAME}
    PRIVATE
      ${tensorflow_LIBRARY}
  )

endfunction(ConfigTarget)