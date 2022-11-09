#include "includes.h"
#include "model.h"

using namespace pycomet;


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

double *pycomet::call_model(
	pycomet::model *mdl,
	double *inputs, 
	std::int32_t *n
)
{
	return mdl->call(inputs, *n);
}
