#include <iostream>
using namespace std;

class model  {

	public:

		int8_t inp_tot_dim = 3;
		int8_t out_tot_dim = 3;

		model(char *filename);

		~model();

        // Call functions
        double *call(double *inputs, int64_t n);

};

model::model(char *filename) {
    std::cout << "c::constr >> model constructed from file " << filename << std::endl;
}

double *model::call(double *inputs, int64_t n) {
    std::cout << "c::call >> inputs = " << *inputs << std::endl;

    double arr[n][inp_tot_dim];
	double *dptr;
	dptr = (double *)inputs;
	for (int i = 0; i <n; i++)
	{
		for (int j = 0; j<inp_tot_dim; j++) {
			arr[i][j] = dptr[j*n+i];
			printf("%.6g ", arr[i][j]);
		}
		printf("\n");
	}

    return inputs;
}

model::~model() {
    std::cout << "c::destr >> model destructed." << std::endl;
}


extern "C" {

	model *init_model(char *filename) {
		return new model(filename);
	}

	int8_t get_inp_tot_dim(model *mdl) {
		return mdl->inp_tot_dim;
	}

	int8_t get_out_tot_dim(model *mdl) {
		return mdl->out_tot_dim;
	}

	void delete_model(model *mdl) {
		mdl->~model();
	}

	double *call_model(model *mdl, double *inputs, int64_t *n) {
    	return mdl->call(inputs, *n);
	}
}


// https://stackoverflow.com/questions/27584674/passing-2d-array-from-fortran-to-c