
#include "pycomet.h"

using namespace pycomet;

inline std::string pycomet::version() {
  return cppflow::version() + " PyCOMET: 1.0";
}
