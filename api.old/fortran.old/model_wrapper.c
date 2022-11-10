#include "pycomet/model.h"

pycomet::model *model_ptr;

extern void bridge(double *);

extern "C" {

	void c_init_model(char *file) {
		model_ptr = new pycomet::model(file);
	}

	void c_delete_model() {
		delete model_ptr;
	}

	void c_call_model(double *inputs) {
		double *outputs = model_ptr->call(inputs);
		bridge(outputs);
		outputs = 0;
	}
}