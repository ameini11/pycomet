#include <fstream>
#include <numeric>
#include <nlohmann/json.hpp>
#include <tensorflow/c/c_api.h>
#include "utils.h"
#include "model.h"

using json = nlohmann::json;


// Constructor
/* ========================================================================= */
pycomet::model::model(const std::string inpfile)
{
  // Parse inputs from json file
  this->parse_inputs(inpfile);
  // Set global TF context
  if (this->config != nullptr)
    this->set_global_context(this->config);
  // Initialize TensorFlow model
  this->tfmodel = std::unique_ptr<cppflow::model>(
    new cppflow::model(this->path_to_model)
  );
  // Get input/output operations
  this->get_ops_info();
  // Check inputs/outputs operations
  this->inputs_dim = this->check_io_ops(this->inputs_id);
  this->outputs_dim = this->check_io_ops(this->outputs_id);
  // Inputs/outputs total dimensions
  this->inp_tot_dim = std::accumulate(
    this->inputs_dim.begin(), this->inputs_dim.end(), 0
  );
  this->out_tot_dim = std::accumulate(
    this->outputs_dim.begin(), this->outputs_dim.end(), 0
  );
}

// Auxiliary functions
/* ========================================================================= */
void pycomet::model::parse_inputs(const std::string inpfile)
{
  // Initialize Json object reader
  std::ifstream file(inpfile);
  json inputs = json::parse(file);
  // Extract inputs
  this->path_to_model = inputs["path_to_model"];
  this->inputs_id = inputs["inputs_id"];
  this->outputs_id = inputs["outputs_id"];
  if (inputs.contains("config"))
  {
    std::vector<std::uint8_t> cfg = inputs["config"];
    this->config = &cfg;
  }
  if (inputs.contains("batch_size"))
  {
    this->batch_size = inputs["batch_size"];
  }
  if (inputs.contains("rowmajor"))
  {
    this->rowmajor = inputs["rowmajor"];
  }
};

void pycomet::model::set_global_context(std::vector<std::uint8_t> *config)
{
  // Create new options with the new configuration
  TFE_ContextOptions *opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetConfig(
    opts, config->data(), config->size(), cppflow::context::get_status()
  );
  // Replace the global context with your options
  cppflow::get_global_context() = cppflow::context(opts);
};

void pycomet::model::get_ops_info()
{
  // Get operations identifiers
  std::vector<std::string> ops_id = this->tfmodel->get_operations();
  // Store identifier-shape pairs
  for (auto const &op_id: ops_id)
  {
    if (op_id.rfind(this->inputs_id_prefix, 0) == 0 ||
      op_id.rfind(this->outputs_id_prefix, 0) == 0)
    {
      auto shape = this->tfmodel->get_operation_shape(op_id);
      this->ops.push_back(std::make_tuple(op_id, shape));
    }
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
      std::cout << "\n Error! Operations identifier '" << i_id;
      std::cout << "' not found!\nAvailable options:" << std::endl;
      for (auto const &op: this->ops)
        std::cout << "\n " << std::get<0>(op) << std::endl;
      exit(1);
    }
  }
  std::vector<std::int32_t> dim_(dim.begin(), dim.end());
  return dim_;
};

// Inputs/Outputs manipulations
/* ========================================================================= */
template <typename T>
std::vector<T> pycomet::model::compose_outputs(
  std::vector<cppflow::tensor> outputs,
  const std::int32_t n
)
{
  std::vector<T> Y;
  for (std::uint16_t i = 0; i < this->outputs_dim.size(); ++i)
  {
    // Extract data from i-th output (row-major ordering)
    auto y = outputs[i].get_data<T>();
    // Eventually reorder i-th output (only if inputs are col-major)
    auto y_ = pycomet::vec_ops::reorder_vec(
      y, n, this->outputs_dim[i], true, this->rowmajor
    );
    // Append inputs
    Y.insert(Y.end(), y_.begin(), y_.end());
  }
  return Y;
}

template <typename T>
std::vector<std::vector<T>> pycomet::model::compose_inputs(
  std::vector<T> inputs,
  const std::int32_t n
)
{
  std::vector<std::vector<T>> X;
  if (this->inputs_dim.size() > 1)
  {
    int start = 0;
    int end = 0;
    int delta;
    for (std::uint16_t i = 0; i < this->inputs_dim.size(); ++i)
    {
      // Extract i-th input
      delta = this->inputs_dim[i] * n;
      end = start + delta;
      std::vector<T> x(inputs.begin() + start, inputs.begin() + end);
      start = end;
      // Eventually reorder i-th input (row-major/col-major ordering)
      std::vector<T> x_ = pycomet::vec_ops::reorder_vec(
        x, n, this->inputs_dim[i], false, this->rowmajor
      );
      // Append the i-th input
      X.push_back(x_);
    }
  }
  else
  {
    // Eventually reorder inputs
    auto inputs_ = pycomet::vec_ops::reorder_vec(
      inputs, n, this->inputs_dim[0], false, this->rowmajor
    );
    // Append inputs
    X.push_back(inputs_);
  }
  return X;
}

template <typename T>
std::vector<std::tuple<std::string, cppflow::tensor>> pycomet::model::convert_inputs(
  std::vector<std::vector<T>> inputs,
  const std::int32_t n
)
{
  std::vector<std::tuple<std::string, cppflow::tensor>> X;
  for (std::uint16_t i = 0; i < this->inputs_dim.size(); ++i)
  {
    // Convert std::vector to cppflow::tensor with the proper shape
    auto x_ten = cppflow::tensor(inputs[i], {n, this->inputs_dim[i]});
    // Make and append the tuple for i-th input
    X.push_back(std::make_tuple(this->inputs_id[i], x_ten));
  };
  return X;
}

// Calling functions
/* ========================================================================= */
template <typename T>
std::vector<T> pycomet::model::call(
  std::vector<T> inputs,
  const std::int32_t n
)
{
  // Inputs manipulation
  std::vector<std::vector<T>> I;
  I = pycomet::model::compose_inputs(inputs, n);
  std::vector<std::tuple<std::string, cppflow::tensor>> X;
  X = pycomet::model::convert_inputs(I, n);
  // Perform inference
  std::vector<cppflow::tensor> outputs = (*this->tfmodel)(X, this->outputs_id);
  // Outputs manipulation
  std::vector<T> Y = pycomet::model::compose_outputs<T>(outputs, n);
  return Y;
}

// template <typename T>
// std::vector<T> pycomet::model::call(
//   const std::vector<T> inputs,
//   const std::int32_t n,
//   const bool rowmajor
// )
// {
//   std::vector<T> I;
//   I.assign(inputs.begin(), inputs.end());
//   // Reshape inputs
//   if ((this->inp_tot_dim > 1) && (rowmajor == true))
//   {
//     auto I_arr = pycomet::vec_ops::reshape(I, {n, this->inp_tot_dim});
//     auto I_vec = pycomet::vec_ops::flatten(I_arr, true);
//     I.assign(I_vec.begin(), I_vec.end());
//   }
//   // Compose inputs
//   std::vector<std::tuple<std::string, cppflow::tensor>> X;
//   int start = 0;
//   int end = 0;
//   int delta;
//   for (std::uint16_t i = 0; i < this->inputs_dim.size(); ++i)
//   {
//     // Extract i-th input
//     delta = this->inputs_dim[i] * n;
//     end = start + delta;
//     std::vector<T> x(I.begin() + start, I.begin() + end);
//     start = end;
//     // Reshape i-th input
//     if (this->inputs_dim[i] > 1)
//     {
//       auto x_arr = pycomet::vec_ops::reshape(x, {this->inputs_dim[i], n});
//       auto x_vec = pycomet::vec_ops::flatten(x_arr, true);
//       x.assign(x_vec.begin(), x_vec.end());
//     }
//     // Convert std::vector to cppflow::tensor with the proper shape
//     cppflow::tensor x_ten = cppflow::tensor(x, {n, this->inputs_dim[i]});
//     // Make and append the tuple for i-th input
//     X.push_back(std::make_tuple(this->inputs_id[i], x_ten));
//   };
//   // Perform inference
//   auto outputs = (*this->tfmodel)(X, this->outputs_id);
//   // Compose outputs
//   std::vector<T> Y;
//   for (std::uint16_t i = 0; i < this->outputs_dim.size(); ++i)
//   {
//     // Extract data from i-th output
//     auto y = outputs[i].get_data<T>();
//     // Reshape i-th output
//     if (this->outputs_dim[i] > 1)
//     {
//       auto y_arr = pycomet::vec_ops::reshape(y, {n, this->outputs_dim[i]});
//       auto y_vec = pycomet::vec_ops::flatten(y_arr, true);
//       y.assign(y_vec.begin(), y_vec.end());
//     }
//     Y.insert(Y.end(), y.begin(), y.end());
//   }
//   // Reshape outputs
//   if ((this->out_tot_dim > 1) && (rowmajor == true))
//   {
//     auto Y_arr = pycomet::vec_ops::reshape(Y, {this->out_tot_dim, n}, true);
//     auto Y_vec = pycomet::vec_ops::flatten(Y_arr);
//     Y.assign(Y_vec.begin(), Y_vec.end());
//   }
//   return Y;
// }

template <typename T>
std::vector<std::vector<T>> pycomet::model::call(
  const std::vector<std::vector<T>> inputs,   // row-major data
  const std::int32_t n
)
{
  // Force col-major
  this->rowmajor = false;
  // Flatten inputs
  std::vector<T> X = pycomet::vec_ops::flatten(inputs, true);
  // Perform inference
  std::vector<T> Y = this->call(X, n);
  // Reshape outputs
  std::vector<std::vector<T>> outputs = pycomet::vec_ops::reshape(
    Y, {this->out_tot_dim, n}, true
  );
  return outputs;
}

// template <typename T>
// std::vector<T> pycomet::model::call(
//   const T *inputs,
//   const std::int32_t n,
//   const bool rowmajor
// )
// {
//   // Convert inputs from array to std::vector
//   std::vector<T> X(inputs, inputs + this->inp_tot_dim*n);
//   // Perform inference
//   std::vector<T> Y = this->call(X, n, rowmajor);
//   return Y;
// }

// Explicit templates instantiation
/* ========================================================================= */
// Single-precision floating-point format -------------------------------------
template std::vector<float> pycomet::model::compose_outputs(
  std::vector<cppflow::tensor> outputs,
  const std::int32_t n
);
template std::vector<std::vector<float>> pycomet::model::compose_inputs(
  std::vector<float> inputs,
  const std::int32_t n
);
template std::vector<std::tuple<std::string, cppflow::tensor>> pycomet::model::convert_inputs(
  std::vector<std::vector<float>> inputs,
  const std::int32_t n
);
template std::vector<float> pycomet::model::call(
  const std::vector<float> inputs,
  const std::int32_t n
);
template std::vector<std::vector<float>> pycomet::model::call(
  const std::vector<std::vector<float>> inputs,
  const std::int32_t n
);
// template std::vector<float> pycomet::model::call(
//   const float *inputs,
//   const std::int32_t n,
//   const bool rowmajor
// );

// Double-precision floating-point format -------------------------------------
template std::vector<double> pycomet::model::compose_outputs(
  std::vector<cppflow::tensor> outputs,
  const std::int32_t n
);
template std::vector<std::vector<double>> pycomet::model::compose_inputs(
  std::vector<double> inputs,
  const std::int32_t n
);
template std::vector<std::tuple<std::string, cppflow::tensor>> pycomet::model::convert_inputs(
  std::vector<std::vector<double>> inputs,
  const std::int32_t n
);
template std::vector<double> pycomet::model::call(
  const std::vector<double> inputs,
  const std::int32_t n
);
template std::vector<std::vector<double>> pycomet::model::call(
  const std::vector<std::vector<double>> inputs,
  const std::int32_t n
);
// template std::vector<double> pycomet::model::call(
//   const double *inputs,
//   const std::int32_t n,
//   const bool rowmajor
// );