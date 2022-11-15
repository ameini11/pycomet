#include "interface.h"


// External bindings
// ========================================================================
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

// Calling functions
// extern "C++"
// {
//   template <typename T>
//   T *pycomet::call_model(
//     pycomet::model *mdl, T *inputs, std::int32_t *n
//   )
//   {
//     return mdl->call(inputs, *n);
//   }

//   template std::int32_t *pycomet::call_model(
//     pycomet::model *mdl, std::int32_t *inputs, std::int32_t *n
//   );

//   template float *pycomet::call_model(
//     pycomet::model *mdl, float *inputs, std::int32_t *n
//   );

//   template double *pycomet::call_model(
//     pycomet::model *mdl, double *inputs, std::int32_t *n
//   );
// }
float *pycomet::call_model(
  pycomet::model *mdl, float *inputs, std::int32_t *n
)
{
  return mdl->call(inputs, *n);
}
