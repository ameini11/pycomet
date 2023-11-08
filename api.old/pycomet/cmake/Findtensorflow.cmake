include(FindPackageHandleStandardArgs)

find_library(tensorflow_LIBRARY
  NAMES tensorflow
)
find_path(tensorflow_INCLUDE_DIR
  NAMES tensorflow/c/c_api.h
)

find_package_handle_standard_args(tensorflow
  REQUIRED_VARS tensorflow_LIBRARY tensorflow_INCLUDE_DIR
  HANDLE_COMPONENTS
)

if(tensorflow_FOUND)
  mark_as_advanced(tensorflow_INCLUDE_DIR)
  mark_as_advanced(tensorflow_LIBRARY)
endif()
