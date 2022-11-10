find_path(cppflow_INCLUDE_DIRS
  NAMES cppflow/cppflow.h
)
mark_as_advanced(cppflow_INCLUDE_DIRS)

find_library(cppflow_LIBRARIES
  NAMES cppflow
)
mark_as_advanced(cppflow_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cppflow
  DEFAULT_MSG
  cppflow_INCLUDE_DIRS
  cppflow_LIBRARIES
)