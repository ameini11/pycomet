find_path(tensorflow_INCLUDE_DIRS
  NAMES tensorflow/c/c_api.h
)
mark_as_advanced(tensorflow_INCLUDE_DIRS)

find_library(tensorflow_LIBRARIES
  NAMES tensorflow
)
mark_as_advanced(tensorflow_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tensorflow
  DEFAULT_MSG
  tensorflow_INCLUDE_DIRS
  tensorflow_LIBRARIES
)
