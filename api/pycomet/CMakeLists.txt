cmake_minimum_required(VERSION 3.10)


# Global
# -----------------------------------------------------------------------------
project(pycomet
  VERSION 1.0
  LANGUAGES C CXX Fortran
  DESCRIPTION "Physics-informed machine learning for scientific computing and operator discovery"
)


# Include cmake modules
# -----------------------------------------------------------------------------
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)
include(EnsureOutOfSourceBuild)
include(LogMessage)
ensure_out_of_source_build()
setLogsOn()


# Set c++ version
# -----------------------------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
endif()
if(CMAKE_CXX_STANDARD LESS 17)
  LogWarning("Incompatible C++ standard ${CMAKE_CXX_STANDARD}. PyCOMET requires C++17")
endif()


# Find external packages
# -----------------------------------------------------------------------------
find_package(tensorflow REQUIRED)
find_package(cppflow REQUIRED)
find_package(json REQUIRED)


# Target
# -----------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED)
file(GLOB_RECURSE SOURCE_FILES_C RELATIVE ${CMAKE_SOURCE_DIR} "src/*.c")
file(GLOB_RECURSE SOURCE_FILES_F RELATIVE ${CMAKE_SOURCE_DIR} "src/*.f90")
target_sources(${PROJECT_NAME}
  PRIVATE
    ${SOURCE_FILES_C}
    ${SOURCE_FILES_F}
)
set_source_files_properties(${SOURCE_FILES_C} PROPERTIES LANGUAGE CXX)


# Include directories
# -----------------------------------------------------------------------------
target_include_directories(${PROJECT_NAME}
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


# Installation path
# -----------------------------------------------------------------------------
# note that it is not CMAKE_INSTALL_PREFIX we are checking here
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  LogMessage("CMAKE_INSTALL_PREFIX is not set")
  LogMessage(">> Default value: ${CMAKE_INSTALL_PREFIX}")
  LogMessage(">> Will set it to ${CMAKE_SOURCE_DIR}/install")
  set(CMAKE_INSTALL_PREFIX
    "${CMAKE_SOURCE_DIR}/install"
    CACHE PATH "Where the library will be installed to" FORCE
  )
else()
  LogMessage("CMAKE_INSTALL_PREFIX is already set")
  LogMessage(">> Current value: ${CMAKE_INSTALL_PREFIX}")
endif()


# Public headers
# -----------------------------------------------------------------------------
# Next thing you need to do is to declare PUBLIC_HEADER property
file(GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_SOURCE_DIR} "include/*.h")
set(PUBLIC_HEADERS ${INCLUDE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")


# Build type
# -----------------------------------------------------------------------------
# Two buid types are available:
# - "debug":    Uses debug compiler options
# - "release":  Uses optimized compiler options
# The default build type is "release".
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
LogMessage("Build type: ${CMAKE_BUILD_TYPE}")


# Debug suffix
# -----------------------------------------------------------------------------
# It might be a good idea to add d suffix to debug binaries - that way you’ll
# get lib<name>.a with Debug configuration and lib<name>.a with Release. To do
# that you need to set the DEBUG_POSTFIX property.
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")


# External libraries
# -----------------------------------------------------------------------------
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    ${tensorflow_LIBRARY}
)


# Destinations
# -----------------------------------------------------------------------------
# Install the target and create export-set
install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Targets"
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


# Build examples
# -----------------------------------------------------------------------------
# option(BUILD_EXAMPLES "Build examples" OFF)
# if(BUILD_EXAMPLES)
#   add_subdirectory(examples)
# endif()

# if (BUILD_APPLICATIONS)
#   set(APPLICATION_NAME ${LIBRARY_NAME}-mergelevels.x)                               # Set the name of the application (ie the executable) to be build
#   set(TARGET_NAME ${APPLICATION_NAME})                                              # Set the name of the target to be build
#   LogMessage("Building the ${APPLICATION_NAME} code")                               # Write log message
#   add_subdirectory(app/MergeLevels)                                                 # Adding the directory containing the source files
#   get_sources(SRCS)                                                                 # Get the list of source files and reseting the global variable
#   add_executable(${TARGET_NAME} ${SRCS})                                            # Create a application code from the list of source files
#   set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${APPLICATION_NAME})  # Change the name of the executable from 'spark-driver' to 'spark'
#   target_link_libraries(${TARGET_NAME} PUBLIC ${STATIC_LIBRARY})                    # Linking the 'spark' executable to the 'spark-cfd' library, which will also link to the 'spark' and 'spark-ode' libraries due to the PUBLIC dependency
#   install(TARGETS ${TARGET_NAME} DESTINATION ${EXECUTABLE_INSTALL_DIR})             # Installing the exectuable
# endif()


# Configs
# -----------------------------------------------------------------------------
# Generate and install export file
install(EXPORT "${PROJECT_NAME}Targets"
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${namespace}::
    DESTINATION cmake
)

include(CMakePackageConfigHelpers)

# Generate the version file for the config file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)
# Create config file
configure_package_config_file(${CMAKE_MODULE_PATH}/${PROJECT_NAME}Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION cmake
)
# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION cmake
)
# Generate the export targets for the build tree
export(EXPORT "${PROJECT_NAME}Targets"
    FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${namespace}::
)
