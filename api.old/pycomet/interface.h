#ifndef pycomet_interface_h_
#define pycomet_interface_h_


#include "includes.h"
#include "model.h"


namespace pycomet
{
	// External bindings
	// ========================================================================
	#ifdef __cplusplus
	extern "C"
	{
	#endif

		pycomet::model *init_model(char *inpfile);

		std::int32_t get_inp_tot_dim(pycomet::model *mdl);

		std::int32_t get_out_tot_dim(pycomet::model *mdl);

		void delete_model(pycomet::model *mdl);

		double *call_model(pycomet::model *mdl, double *inputs, std::int32_t *n);

	#ifdef __cplusplus
	}
	#endif
}


#endif // pycomet_interface_h_
