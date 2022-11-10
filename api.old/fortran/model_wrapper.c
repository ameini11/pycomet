// #include "pycomet/model.h"

// extern "C" {

// 	pycomet::model *init_model(char *file) {
// 		return new pycomet::model(file);
// 	}

// 	void delete_model(pycomet::model *model) {
// 		delete model;
// 	}

// 	double call_model(pycomet::model *model, double *inputs) {
//     	return model->call(inputs);
// 	}
// }





// crectangle_class.cc
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
    std::cout << "Model constructed: " << filename << std::endl;
}

double model::call(double *inputs) {
    std::cout << "c inputs: " << *inputs << std::endl;
    return *inputs;
}

model::~model() {
    std::cout << "Model deconstructed: " << std::endl;
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




// // example main() written in C++:
// int main () {
	
// 	char filename = '1';
// 	double inp = 0.1;
// 	double out;

// 	model mdl = model(&filename);

// 	out = mdl.call(inp);

//     std::cout << "c outputs: " << out << std::endl;

// 	return 0;
// }

