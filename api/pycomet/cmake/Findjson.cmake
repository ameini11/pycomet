find_package(PkgConfig)

# The user can specify the include directory manually
if(NOT EXISTS "${json_INCLUDE_DIR}")
  find_path(json_INCLUDE_DIR
    NAMES nlohmann/json.hpp
  )
endif()

if(EXISTS "${json_INCLUDE_DIR}")
  include(FindPackageHandleStandardArgs)
  mark_as_advanced(json_INCLUDE_DIR)
else()
  include(ExternalProject)
  ExternalProject_Add(json
    GIT_REPOSITORY https://github.com/nlohmann/json
    TIMEOUT 5
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third-party/json
    CMAKE_ARGS
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  )
  # Specify include directory
  set(json_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()

if(EXISTS ${json_INCLUDE_DIR})
  set(json_FOUND 1)
else()
  set(json_FOUND 0)
endif()
