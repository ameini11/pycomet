#ifndef pycomet_interface_h_
#define pycomet_interface_h_


#include "includes.h"
#include "model.h"


namespace pycomet
{
  // External bindings
  // ==================================
  #ifdef __cplusplus
  extern "C" {
  #endif // __cplusplus

    pycomet::model* init_model(char* inpfile);

    std::int32_t get_inp_tot_dim(pycomet::model* mdl);

    std::int32_t get_out_tot_dim(pycomet::model* mdl);

    void delete_model(pycomet::model* mdl);

    void call_model_float(
      pycomet::model* mdl,      // PyCOMET model
      std::int32_t* n,          // Total number of evaluated points
      float* inputs,            // 1D array of inputs (row-/column-major)
      float* outputs            // Empty 1D array of outputs
    );

    void call_model_double(
      pycomet::model* mdl,      // PyCOMET model
      std::int32_t* n,          // Total number of evaluated points
      double* inputs,           // 1D array of inputs (row-/column-major)
      double* outputs           // Empty 1D array of outputs
    );

  #ifdef __cplusplus
  } // extern "C"
  #endif // __cplusplus
}


#endif // pycomet_interface_h_
