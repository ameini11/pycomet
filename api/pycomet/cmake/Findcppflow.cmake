# find_package(PkgConfig)

# # The user can specify the include directory manually
# if(NOT EXISTS "${cppflow_INCLUDE_DIR}")
#   find_path(cppflow_INCLUDE_DIR
#     NAMES cppflow/cppflow.h
#   )
# endif()

# if(EXISTS "${cppflow_INCLUDE_DIR}")
#   include(FindPackageHandleStandardArgs)
#   mark_as_advanced(cppflow_INCLUDE_DIR)
# else()
#   include(ExternalProject)
#   ExternalProject_Add(cppflow
#     GIT_REPOSITORY https://github.com/serizba/cppflow
#     TIMEOUT 5
#     CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
#     PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
#     INSTALL_COMMAND "" # Disable install step, this is a header-only library
#     BUILD_COMMAND "" # Disable build step, this is a header-only library
#   )
#   # Specify include directory
#   ExternalProject_Get_Property(cppflow source_dir)
#   set(cppflow_INCLUDE_DIRS ${source_dir}/include)
# endif()

# if(EXISTS "${cppflow_INCLUDE_DIR}")
#   set(cppflow_FOUND 1)
# else()
#   set(cppflow_FOUND 0)
# endif()

find_package(PkgConfig)

include(ExternalProject)

# Set the path to the external project source directory
set(cppflow_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third-party/cppflow)

# Add the cppflow project using ExternalProject_Add
ExternalProject_Add(cppflow
  SOURCE_DIR ${cppflow_SOURCE_DIR}
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/cppflow"
  INSTALL_COMMAND ""  # Disable install step, as this is a header-only library
  BUILD_COMMAND ""    # Disable build step, as this is a header-only library
)

# Set cppflow_INCLUDE_DIR based on the source directory
set(cppflow_INCLUDE_DIR "${cppflow_SOURCE_DIR}/include")

# Set cppflow_FOUND based on the existence of the include directory
if(EXISTS "${cppflow_INCLUDE_DIR}")
  set(cppflow_FOUND 1)
else()
  set(cppflow_FOUND 0)
endif()
