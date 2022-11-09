#include <cppflow/cppflow.h>
#include "pycomet/pycomet.h"


inline std::string pycomet::version() {
  return cppflow::version() + " PyCOMET: 1.0";
}
