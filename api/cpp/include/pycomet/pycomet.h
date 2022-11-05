#ifndef INCLUDE_PYCOMET_PYCOMET_H_
#define INCLUDE_PYCOMET_PYCOMET_H_

// C++ headers
#include <string>

// CppFlow headers
#include "cppflow/cppflow.h"

// PyCOMET headers
// #include "pycomet/model.h"


namespace pycomet {

  // Version of TensorFlow, CppFlow, and PyCOMET
  std::string version();

}

namespace pycomet {

  inline std::string version() {
    return cppflow::version() + " PyCOMET: 1.0";
  }

}


#endif
