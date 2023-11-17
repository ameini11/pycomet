#ifndef pycomet_pycomet_h_
#define pycomet_pycomet_h_

// PyCOMET headers
#include "includes.h"  // Include necessary headers for the project
#include "utils.h"     // Utility functions
#include "model.h"     // Machine learning model-related functionality
#include "interface.h" // Interface-related functionality

namespace pycomet {

  /**
   * @brief Retrieves the version information for TensorFlow, CppFlow, and PyCOMET.
   *
   * This function returns a string representing the version information of
   * TensorFlow, CppFlow, and the PyCOMET library.
   *
   * @return A string containing the version information.
   */
  std::string version();

} // namespace pycomet

#endif // pycomet_pycomet_h_
