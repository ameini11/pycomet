#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <map>

#include "cppflow"


using namespace std;

/* ------------------------------------------------------------------------- */
// Model class declaration
/* ------------------------------------------------------------------------- */
class model {

	private:

	    // TensorFlow model
	    cppflow::model tf_model;

	    // Inputs/Outputs
		const std::vector<std::string> inp_id;
		const std::vector<std::string> out_id;
		const std::vector<std::int8_t> inp_dim;
		const std::vector<std::int8_t> out_dim;

	    // Inputs/Outputs prefix identifier
	    const std::string inp_id_prefix = 'serving_default';
	    const std::string out_id_prefix = 'StatefulPartitionedCall';

	public:

	    // Constructor
	    explicit model(const std::string inp_file);

	    // Destructor
		~model() = default;

	    // Inputs/Outputs total dimensions
		const std::int8_t inp_tot_dim;
		const std::int8_t out_tot_dim;

	    // Call functions
	    template<T>
	    std::vector<std::vector<T>> call(std::vector<std::vector<T>> inputs, std::int64_t n);

	    template<T>
	    std::vector<T> call(std::vector<T> inputs, std::int64_t n);

	    template<T>
        T *call(T *inputs, std::int64_t n);

};

/* ------------------------------------------------------------------------- */
// Model class definition
/* ------------------------------------------------------------------------- */
inline model::model(const std::string inp_file) {

	// std::string path_to_model = read from inputs
	std::string path_to_model = inp_file;

    // Initialize the model
	this->tf_model = cppflow::model(path_to_model);

    // Get input/output identifiers
	// this->inp_id = read from inputs
	// this->out_id = read from inputs
	this->inp_id = {"serving_default_input_1:0", "serving_default_input_2:0"};
	this->out_id = {"StatefulPartitionedCall:0"};

    for (auto const &i: this->inp_id) {
		if(i.substr(0, this->inp_id_prefix.size()) != this->inp_id_prefix) {
		    std::cout << "\n Error! Inputs identifier mismatch!";
		    exit(1);
		}
    }
    for (auto const &o: this->out_id) {
		if(o.substr(0, this->out_id_prefix.size()) != this->out_id_prefix) {
		    std::cout << "\n Error! Outputs identifier mismatch!";
		    exit(1);
		}
    }

    // Get input/output dimensions
	// this->inp_dim = read from inputs
	// this->out_dim = read from inputs
	this->inp_dim = {2, 1};
	this->out_dim = {2};

	if(this->inp_dim.size() != this->inp_id.size()) {
	    std::cout << "\n Error! mismatch between inputs dimensions and identifiers";
	    exit(1);
	}
	if(this->out_dim.size() != this->out_id.size()) {
	    std::cout << "\n Error! mismatch between outputs dimensions and identifiers";
	    exit(1);
	}

	// Inputs/Outputs total dimensions
	this->inp_tot_dim = std::accumulate(this->inp_dim.begin(),this->inp_dim.end(),0);
	this->out_tot_dim = std::accumulate(this->out_dim.begin(),this->out_dim.end(),0);

};

// Call functions
template<T>
std::vector<T> model::call(std::vector<T> inputs, std::int64_t n) {

	// Compose inputs ---------------------------------------------------------
	std::vector<std::tuple<std::string, cppflow::tensor>> X;
	int start = 0;
	int end = 0;
	int delta;
	for (int i = 0; i < this->inp_dim.size(); ++i) {
		// Extract input i
		delta = this->inp_dim[i] * n
		end = start + delta
		std::vector<T> x_vec(inputs.begin() + start, inputs.begin() + end);
		start = end
		// Convert std::vector to cppflow::tensor
    	cppflow::tensor x_ten = cppflow::tensor(x_vec, {n, this->inp_dim[i]});
		// Make and append the tuple for input i
		X.push_back(std::make_tuple(this->inp_id[i], x_ten));
	}

	// Perform inference ------------------------------------------------------
    auto outputs = this->tf_model(X, this->out_id);

	// Compose outputs --------------------------------------------------------
	std::vector<T> Y;
	for (int i = 0; i < this->out_dim.size(); ++i) {
		auto y = output[i].get_data<T>();
		Y.insert(Y.end(), y.begin(), y.end());
	}

	return Y

};

template<T>
std::vector<std::vector<T>> model::call(std::vector<std::vector<T>> inputs, std::int64_t n) {

	auto X = utils::flatten(inputs)

	auto Y = this->call(X, n)

	auto shape = std::make_tuple(n, this->out_tot_dim)
	auto outputs = utils::reshape(Y, shape)

	return outputs

};

template<T>
T *model::call(T *inputs, std::int64_t n) {

	std::vector<T> X(inputs, this->inp_tot_dim*n);

	auto Y = this->call(X, n)

	T *outputs = &Y[0];

	return outputs

};


/* ------------------------------------------------------------------------- */
// Fortran binding
/* ------------------------------------------------------------------------- */
extern "C" {

	model *init_model(char *inp_file) {
		std::string file(inp_file);
		return new model(file);
	}

	std::int8_t get_inp_tot_dim(model *mdl) {
		return mdl->inp_tot_dim;
	}

	std::int8_t get_out_tot_dim(model *mdl) {
		return mdl->out_tot_dim;
	}

	void delete_model(model *mdl) {
		mdl->~model();
	}

	double *call_model(model *mdl, double *inputs, std::int64_t *n) {
    	return mdl->call(inputs, *n);
	}
}
