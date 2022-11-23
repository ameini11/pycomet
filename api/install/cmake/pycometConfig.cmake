set(pycomet_FOUND 1)

set(pycomet_INCLUDE_DIR "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/install/include")
set(pycomet_INCLUDE_DIRS ${pycomet_INCLUDE_DIR})

set(pycomet_LIBRARY_DIR "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/install/lib")

find_library(pycomet_LIBRARY pycomet HINTS ${pycomet_LIBRARY_DIR})

set(pycomet_LIBRARIES ${pycomet_LIBRARY})
