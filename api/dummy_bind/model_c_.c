#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <map>

#include "cppflow"
#include <json/json.h>


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
		std::vector<std::int64_t> model::check_io_ops(
			std::vector<std::string> identifiers
		)

	    // IO operations
	    // const std::string inp_id_prefix = 'serving_default';
	    // const std::string out_id_prefix = 'StatefulPartitionedCall';
	    const std::vector<std::tuple<std::string, std::vector<int64_t>>> ops;

	    void get_ops_info();

	    // Input parser
	    void parse_inputs(std::string inp_file)

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
	    std::vector<std::vector<T>> call(
	    	std::vector<std::vector<T>> inputs, std::int64_t n
	    );

	    template<T>
	    std::vector<T> call(std::vector<T> inputs, std::int64_t n);

	    template<T>
        T *call(T *inputs, std::int64_t n);

};

/* ------------------------------------------------------------------------- */
// Model class definition
/* ------------------------------------------------------------------------- */
inline model::model(const std::string inp_file)
{
	// Parse inputs from Json file
	this->parse_inputs(inp_file);
    // Initialize TensorFlow model
	this->tf_model = cppflow::model(this->path_to_model);
    // Get input/output operations
	this->get_ops_info();
    // Checking inputs/outputs
    this->inp_dim = this->check_io_ops(this->inp_id);
    this->out_dim = this->check_io_ops(this->out_id);
	// Inputs/outputs total dimensions
	this->inp_tot_dim = std::accumulate(
		this->inp_dim.begin(), this->inp_dim.end(), 0
	);
	this->out_tot_dim = std::accumulate(
		this->out_dim.begin(), this->out_dim.end(), 0
	);
};

inline void model::parse_inputs(const std::string inp_file)
{
	// Initialize Json object reader
	Json::Value inputs;
    std::ifstream file(inp_file, std::ifstream::binary);
    file >> inputs;
	// Extract inputs
    this->path_to_model = inputs["path_to_model"];
    this->inp_id = inputs["inp_id"];
    this->out_id = inputs["out_id"];
	// this->inp_id = {"serving_default_input_1:0", "serving_default_input_2:0"};
	// this->out_id = {"StatefulPartitionedCall:0"};
	// this->inp_dim = {2, 1};
	// this->out_dim = {2};
};

inline void model::get_ops_info()
{
    // Get operations identifiers
    std::vector<std::string> ops_id = this->tf_model.get_operations();
    // Store identifier-shape pairs
    for (auto const &op_id: ops_id)
    {
    	std::vector<int64_t> shape = this->tf_model.get_operation_shape(op);
		this->ops.push_back(std::make_tuple(op_id, shape));
    };
};

inline std::vector<std::int64_t> model::check_io_ops(
	std::vector<std::string> identifiers
)
{
	std::vector<std::int64_t> dim;
    for (auto const &i_id: identifiers)
    {
    	bool found = false;
	    for (auto const &op: this->ops)
	    {
	    	auto op_id = std::get<0>(op);
	    	if (op_id == i_id) {
	    		found = true;
	    		dim.push_back(std::get<1>(op)[1]);
	    		break;
	    	};
	    };
		if(found == false)
		{
		    std::cout << "\n Error! Operations identifier '" << i_id << "' not found!" << std::endl;
		    std::cout << "\n Available options:" << std::endl;
		    for (auto const &op: this->ops)
		    	std::cout << "\n " << std::get<0>(op) << std::endl;
		    exit(1);
		};
    };
    return dim;
};

template<T>
std::vector<T> model::call(std::vector<T> inputs, std::int64_t n)
{
	// Compose inputs
	std::vector<std::tuple<std::string, cppflow::tensor>> X;
	int start = 0;
	int end = 0;
	int delta;
	for (int i = 0; i < this->inp_dim.size(); ++i)
	{
		// Extract input i
		delta = this->inp_dim[i] * n;
		end = start + delta;
		std::vector<T> x_vec(inputs.begin() + start, inputs.begin() + end);
		start = end;
		// Convert std::vector to cppflow::tensor
    	cppflow::tensor x_ten = cppflow::tensor(x_vec, {n, this->inp_dim[i]});
		// Make and append the tuple for input i
		X.push_back(std::make_tuple(this->inp_id[i], x_ten));
	};
	// Perform inference
    auto outputs = this->tf_model(X, this->out_id);
	// Compose outputs
	std::vector<T> Y;
	for (int i = 0; i < this->out_dim.size(); ++i)
	{
		auto y = output[i].get_data<T>();
		Y.insert(Y.end(), y.begin(), y.end());
	}
	return Y;
};

template<T>
std::vector<std::vector<T>> model::call(
	std::vector<std::vector<T>> inputs, std::int64_t n
)
{
	// Flatten inputs
	auto X = utils::flatten(inputs);
	// Perform inference
	auto Y = this->call(X, n);
	// Reshape outputs
	auto shape = std::make_tuple(n, this->out_tot_dim);
	auto outputs = utils::reshape(Y, shape);
	return outputs;
};

template<T>
T *model::call(T *inputs, std::int64_t n)
{
	// Convert inputs from array to std::vector
	std::vector<T> X(inputs, this->inp_tot_dim*n);
	// Perform inference
	auto Y = this->call(X, n);
	// Convert outputs from std::vector to array
	T *outputs = &Y[0];
	return outputs;
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
