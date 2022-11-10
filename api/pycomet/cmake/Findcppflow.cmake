find_package(PkgConfig)

# Allow the user can specify the include directory manually
if(NOT EXISTS "${cppflow_INCLUDE_DIR}")
    find_path(cppflow_INCLUDE_DIR
        NAMES cppflow/cppflow.h
    )
endif()

if(EXISTS "${cppflow_INCLUDE_DIR}")
    include(FindPackageHandleStandardArgs)
    mark_as_advanced(cppflow_INCLUDE_DIR)
else()
    include(ExternalProject)
    ExternalProject_Add(cppflow
        GIT_REPOSITORY https://github.com/serizba/cppflow
        TIMEOUT 5
        CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
        INSTALL_COMMAND "" # Disable install step, this is a header-only library
        BUILD_COMMAND "" # Disable build step, this is a header-only library
    )
    # Specify include directory
    ExternalProject_Get_Property(cppflow source_dir)
    set(cppflow_INCLUDE_DIRS ${source_dir}/include)
endif()

if(EXISTS "${cppflow_INCLUDE_DIR}")
    set(cppflow_FOUND 1)
else()
    set(cppflow_FOUND 0)
endif()