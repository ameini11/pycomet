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
