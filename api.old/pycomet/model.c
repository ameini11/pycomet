#include <fstream>
#include <numeric>
#include <nlohmann/json.hpp>
#include "model.h"

using json = nlohmann::json;


// Model class
// ============================================================================
// Constructor ----------------------------------------------------------------
pycomet::model::model(const std::string inpfile)
{
	// Parse inputs from json file
	this->parse_inputs(inpfile);
    // Initialize TensorFlow model
    this->tfmodel = std::unique_ptr<cppflow::model>(
    	new cppflow::model(this->path_to_model)
    );
    // Get input/output operations
	this->get_ops_info();
    // Check inputs/outputs operations
    this->inp_dim = this->check_io_ops(this->inp_id);
    this->out_dim = this->check_io_ops(this->out_id);
	// Inputs/outputs total dimensions
	this->inp_tot_dim = std::accumulate(
		this->inp_dim.begin(), this->inp_dim.end(), 0
	);
	this->out_tot_dim = std::accumulate(
		this->out_dim.begin(), this->out_dim.end(), 0
	);
}

// Auxiliary functions --------------------------------------------------------
void pycomet::model::parse_inputs(const std::string inpfile)
{
	// Initialize Json object reader
    std::ifstream file(inpfile);
    json inputs = json::parse(file);
	// Extract inputs
    this->path_to_model = inputs["path_to_model"];
    this->inp_id = inputs["inp_id"];
    this->out_id = inputs["out_id"];
};

void pycomet::model::get_ops_info()
{
    // Get operations identifiers
    std::vector<std::string> ops_id = this->tfmodel->get_operations();
    // Store identifier-shape pairs
    for (auto const &op_id: ops_id)
    {
    	auto shape = this->tfmodel->get_operation_shape(op_id);
		this->ops.push_back(std::make_tuple(op_id, shape));
    }
}

std::vector<std::int32_t> pycomet::model::check_io_ops(
	const std::vector<std::string> identifiers
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
	    	}
	    }
		if(found == false)
		{
		    std::cout << "\n Error! Operations identifier '" << i_id << "' not found!" << std::endl;
		    std::cout << "\n Available options:" << std::endl;
		    for (auto const &op: this->ops)
		    	std::cout << "\n " << std::get<0>(op) << std::endl;
		    exit(1);
		}
    }
    std::vector<std::int32_t> dim_(dim.begin(), dim.end());
    return dim_;
};

// Calling functions ----------------------------------------------------------
template <typename T>
std::vector<T> pycomet::model::call(
	std::vector<T> inputs,
	const std::int32_t n
)
{
	// Compose inputs
	std::vector<std::tuple<std::string, cppflow::tensor>> X;
	int start = 0;
	int end = 0;
	int delta;
	for (std::uint16_t i = 0; i < this->inp_dim.size(); ++i)
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
    auto outputs = (*this->tfmodel)(X, this->out_id);
	// Compose outputs
	std::vector<T> Y;
	for (std::uint16_t i = 0; i < this->out_dim.size(); ++i)
	{
		auto y = outputs[i].get_data<T>();
		Y.insert(Y.end(), y.begin(), y.end());
	}
	return Y;
}

template <typename T>
std::vector<std::vector<T>> pycomet::model::call(
	std::vector<std::vector<T>> inputs,
	const std::int32_t n,
    const bool transpose_in,
    const bool transpose_out
)
{
	// Flatten inputs
	auto X = pycomet::flatten(inputs, transpose_in);
	// Perform inference
	auto Y = this->call(X, n);
	// Reshape outputs
	auto outputs = pycomet::reshape(Y, {n, this->out_tot_dim}, transpose_out);
	return outputs;
}

template <typename T>
T *pycomet::model::call(
	T *inputs,
	const std::int32_t n
)
{
	// Convert inputs from array to std::vector
	std::vector<T> X(*inputs, this->inp_tot_dim*n);
	// Perform inference
	auto Y = this->call(X, n);
	// Convert outputs from std::vector to array
	T *outputs = &Y[0];
	return outputs;
}

// Explicit templates instantiation -------------------------------------------
template std::vector<double> pycomet::model::call(
	std::vector<double> inputs, const std::int32_t n
);
template std::vector<std::vector<double>> pycomet::model::call(
	std::vector<std::vector<double>> inputs, const std::int32_t n
);
template double *pycomet::model::call(
	double *inputs, const std::int32_t n
);
