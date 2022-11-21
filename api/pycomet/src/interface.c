#include "interface.h"


// External bindings
/* ========================================================================= */
pycomet::model *pycomet::init_model(char *inpfile)
{
  std::string file(inpfile);
  return new pycomet::model(file);
}

std::int32_t pycomet::get_inp_tot_dim(pycomet::model *mdl)
{
  return mdl->inp_tot_dim;
}

std::int32_t pycomet::get_out_tot_dim(pycomet::model *mdl)
{
  return mdl->out_tot_dim;
}

void pycomet::delete_model(pycomet::model *mdl)
{
  mdl->~model();
}

void pycomet::call_model(
  pycomet::model *mdl,
  std::int32_t *n,
  float *inputs,
  float *outputs
)
{
  // Convert inputs from array to std::vector
  std::vector<float> X(inputs, inputs + mdl->inp_tot_dim*(*n));
  // Perform inference
  std::vector<float> Y = mdl->call(X, *n);
  // Convert outputs from std::vector to array
  std::copy(Y.begin(), Y.end(), outputs);
}

// extern "C++"
// {
//   template <typename T>
//   void pycomet::call_model(
//     pycomet::model *mdl, T *inputs, std::int32_t *n
//   )
//   {
//     // Perform inference
//     std::vector<T> Y = mdl->call(inputs, *n);
//     // Convert outputs from std::vector to array
//     std::copy(Y.begin(), Y.end(), outputs);
//   }

//   void pycomet::call_model(
//     pycomet::model *mdl, std::int32_t *inputs, std::int32_t *n
//   );

//   void pycomet::call_model(
//     pycomet::model *mdl, float *inputs, std::int32_t *n
//   );

//   void pycomet::call_model(
//     pycomet::model *mdl, double *inputs, std::int32_t *n
//   );
// }
