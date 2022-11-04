#include <iostream>
using namespace std;

class model  {

	public:

		model(char *filename);

		~model();

        // Call functions
        double call(double *inputs);

};

model::model (char *filename) {
    std::cout << "c::constr >> model constructed from file " << filename << std::endl;
}

double model::call(double *inputs) {
    std::cout << "c::call >> inputs = " << *inputs << std::endl;
    return *inputs;
}

model::~model() {
    std::cout << "c::destr >> model destructed." << std::endl;
}


extern "C" {

	model *init_model(char *filename) {
		return new model(filename);
	}

	void delete_model(model *mdl) {
		mdl->~model();
	}

	double call_model(model *mdl, double *inputs) {
    	return mdl->call(inputs);
	}
}
