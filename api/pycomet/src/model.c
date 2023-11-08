#include <fstream>
#include <numeric>
#include <nlohmann/json.hpp>
#include <tensorflow/c/c_api.h>
#include "utils.h"
#include "model.h"

using json = nlohmann::json;


// Constructor
// ====================================
// Main function
// ------------------------------------
pycomet::model::model(
  const std::string inpfile
) {
  // Parse inputs from json file
  this->parse_inputs(inpfile);
  // Set global TF context
  if (this->config != nullptr) {
    this->set_global_context(*this->config);
  }
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

// Util functions
// ------------------------------------
void pycomet::model::parse_inputs(
  const std::string inpfile
) {
  // Initialize JSON object reader
  std::ifstream file(inpfile);
  json inputs = json::parse(file);
  // Extract inputs
  this->path_to_model = inputs["path_to_model"];
  this->inputs_id = inputs["inputs_id"];
  this->outputs_id = inputs["outputs_id"];
  this->batch_size = inputs.value("batch_size", this->batch_size);
  this->rowmajor = inputs.value("rowmajor", this->rowmajor);
  if (inputs.contains("config")) {
    std::vector<std::string> cfg = inputs["config"];
    this->config = &cfg;
  }
};

void pycomet::model::set_global_context(
  const std::vector<std::string>& config
) {
  // Create new options with the new configuration
  TFE_ContextOptions *opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetConfig(
    opts, config.data(), config.size(), cppflow::context::get_status()
  );
  // Replace the global context with user options
  cppflow::get_global_context() = cppflow::context(opts);
};

void pycomet::model::get_ops_info() {
  // Get operations identifiers
  std::vector<std::string> ops_id = this->tfmodel->get_operations();
  // Store identifier-shape pairs
  for (const auto &op_id : ops_id) {
    bool c1 = (op_id.rfind(this->inputs_id_prefix, 0) == 0);
    bool c2 = (op_id.rfind(this->outputs_id_prefix, 0) == 0);
    if (c1 || c2) {
      auto shape = this->tfmodel->get_operation_shape(op_id);
      this->ops.emplace_back(op_id, shape);
    }
  }
}

std::vector<std::int32_t> pycomet::model::check_io_ops(
  const std::vector<std::string> identifiers
) {
  std::vector<std::int64_t> dim;
  for (const auto &i_id : identifiers) {
    bool found = false;
    for (const auto &op : this->ops) {
      const std::string op_id = std::get<0>(op);
      if (i_id.find(op_id) != std::string::npos) {
        for (auto i = 0; i < this->max_nb_io; ++i) {
          if (i_id == op_id + ":" + std::to_string(i)) {
            dim.push_back(std::get<1>(op)[1]);
            found = true;
            break;
          }
        }
      }
      if (found) break;
    }
    if (!found) {
      std::ostringstream message;
      message << "\nFrom pycomet::model::check_io_ops():"
              << "\n> Operation identifier '" << i_id << "' not found!"
              << "\n> Available options are:";
      for (const auto &op : this->ops) {
        message << "\n> - " << std::get<0>(op) << ":<i>";
      }
      throw std::runtime_error(message.str());
    }
  }
  // Recast
  std::vector<std::int32_t> _dim(dim.begin(), dim.end());
  return _dim;
};

// Inputs/Outputs manipulations
// ====================================
template <typename T>
std::vector<std::vector<T>> pycomet::model::compose_inputs(
  const std::vector<T>& inputs,
  const std::int32_t nb_pts
) {
  // Get the number of inputs (single-/multi-inputs)
  const size_t nb_inp = this->inputs_dim.size();
  std::vector<std::vector<T>> x(nb_inp);
  // Loop over inputs
  std::int32_t delta, start = 0;
  for (size_t i = 0; i < nb_inp; ++i) {
    // > Extract i-th input
    delta = this->inputs_dim[i] * nb_pts;
    std::vector<T> xi(inputs.begin() + start, inputs.begin() + start + delta);
    // > If column-major, reorder the input
    if (!this->rowmajor) {
      pycomet::ops::invert_major<T>(xi, this->inputs_dim[i], nb_pts, true);
    }
    // > Store the i-th input as a view into the original input vector
    x[i] = std::move(xi);
    // > Update the counter
    start += delta;
  }
  return x;
}

template <typename T>
std::vector<std::tuple<std::string, cppflow::tensor>> pycomet::model::convert_inputs(
  const std::vector<std::vector<T>>& inputs,
  const std::int32_t nb_pts
) {
  // Get the number of inputs (single-/multi-inputs)
  const size_t nb_inp = this->inputs_dim.size();
  std::vector<std::tuple<std::string, cppflow::tensor>> x;//(nb_inp);
  // Loop over inputs
  auto dtype = cppflow::deduce_tf_type<T>();
  for (size_t i = 0; i < nb_inp; ++i) {
    // Convert std::vector to cppflow::tensor with the proper shape
    cppflow::tensor tf_xi(inputs[i], {nb_pts, this->inputs_dim[i]});
    // Make and append the tuple for i-th input
    x.emplace_back(this->inputs_id[i], std::move(tf_xi));
  }
  return x;
}

template <typename T>
void pycomet::model::compose_outputs(
  std::vector<T>& outputs,
  const std::vector<cppflow::tensor>& tf_outputs,
  const std::int32_t nb_pts
) {
  // Get the number of outputs (single-/multi-outputs)
  const size_t nb_out = this->outputs_dim.size();
  // Loop over outputs
  size_t index = 0;
  for (size_t i = 0; i < nb_out; ++i) {
    // Extract data from i-th output (row-major ordering)
    std::vector<T> yi = tf_outputs[i].get_data<T>();
    // > If column-major, reorder the input
    if (!this->rowmajor) {
      pycomet::ops::invert_major<T>(yi, nb_pts, this->outputs_dim[i], false);
    }
    // Move to outputs
    std::move(yi.begin(), yi.end(), outputs.begin() + index);
    index += yi.size();
  }
}

// Calling
// ====================================
template <typename T>
void pycomet::model::call(
  const std::vector<T>& inputs,
  std::vector<T>& outputs,
  const std::int32_t nb_pts
) {
  // Inputs manipulation
  auto x = pycomet::model::compose_inputs<T>(inputs, nb_pts);
  auto tf_x = pycomet::model::convert_inputs<T>(x, nb_pts);
  // Perform inference
  auto tf_y = (*this->tfmodel)(tf_x, this->outputs_id);
  // Outputs manipulation
  pycomet::model::compose_outputs<T>(outputs, tf_y, nb_pts);
}

template <typename T>
std::vector<T> pycomet::model::call(
  const std::vector<T>& inputs,
  const std::int32_t nb_pts
) {
  std::vector<T> outputs(nb_pts * this->out_tot_dim);
  this->call<T>(inputs, outputs, nb_pts);
  return outputs;
}

template <typename T>
std::vector<std::vector<T>> pycomet::model::call(
  const std::vector<std::vector<T>>& inputs,
  const std::int32_t nb_pts
) {
  // Force column-major order
  this->rowmajor = false;
  // Flatten inputs
  auto x = pycomet::ops::flatten<T>(inputs, true);
  // Perform inference
  auto y = this->call<T>(x, nb_pts);
  // Reshape outputs
  std::vector<std::int32_t> shape = {nb_pts, this->out_tot_dim};
  return pycomet::ops::reshape<T>(y, shape, true);
}

// template <typename T>
// std::vector<T> pycomet::model::call(
//   std::vector<T> inputs,
//   const std::int32_t nb_pts
// )
// {
//   // Inputs manipulation
//   std::clock_t c_start = std::clock();
//   std::vector<std::vector<T>> I;
//   std::clock_t c_end = std::clock();
//   long double time = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
//   std::cout << "  Create vector I:        " << time << " milliseconds" << std::endl;

//   c_start = std::clock();
//   I = pycomet::model::compose_inputs(inputs, nb_pts);
//   c_end = std::clock();
//   time = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
//   std::cout << "  Call 'compose_inputs':  " << time << " milliseconds" << std::endl;

//   c_start = std::clock();
//   std::vector<std::tuple<std::string, cppflow::tensor>> X;
//   c_end = std::clock();
//   time = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
//   std::cout << "  Create vector X:        " << time << " milliseconds" << std::endl;

//   c_start = std::clock();
//   X = pycomet::model::convert_inputs(I, nb_pts);
//   c_end = std::clock();
//   time = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
//   std::cout << "  Call 'convert_inputs':  " << time << " milliseconds" << std::endl;

//   // Perform inference
//   c_start = std::clock();
//   std::vector<cppflow::tensor> outputs = (*this->tfmodel)(X, this->outputs_id);
//   c_end = std::clock();
//   time = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
//   std::cout << "  Perform inference:      " << time << " milliseconds" << std::endl;

//   // Outputs manipulation
//   c_start = std::clock();
//   std::vector<T> Y = pycomet::model::compose_outputs<T>(outputs, nb_pts);
//   c_end = std::clock();
//   time = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
//   std::cout << "  Call 'compose_outputs': " << time << " milliseconds" << std::endl;
//   return Y;
// }

// Explicit templates instantiation
// ====================================
// Single-precision floating-point format
// ------------------------------------
template std::vector<std::vector<float>> pycomet::model::compose_inputs(
  const std::vector<float>& inputs,
  const std::int32_t nb_pts
);

template std::vector<std::tuple<std::string, cppflow::tensor>> pycomet::model::convert_inputs(
  const std::vector<std::vector<float>>& inputs,
  const std::int32_t nb_pts
);

template void pycomet::model::compose_outputs(
  std::vector<float>& outputs,
  const std::vector<cppflow::tensor>& tf_outputs,
  const std::int32_t nb_pts
);

template void pycomet::model::call(
  const std::vector<float>& inputs,
  std::vector<float>& outputs,
  const std::int32_t nb_pts
);

template std::vector<float> pycomet::model::call(
  const std::vector<float>& inputs,
  const std::int32_t nb_pts
);

template std::vector<std::vector<float>> pycomet::model::call(
  const std::vector<std::vector<float>>& inputs,
  const std::int32_t nb_pts
);

// Double-precision floating-point format
// ------------------------------------
template std::vector<std::vector<double>> pycomet::model::compose_inputs(
  const std::vector<double>& inputs,
  const std::int32_t nb_pts
);

template std::vector<std::tuple<std::string, cppflow::tensor>> pycomet::model::convert_inputs(
  const std::vector<std::vector<double>>& inputs,
  const std::int32_t nb_pts
);

template void pycomet::model::compose_outputs(
  std::vector<double>& outputs,
  const std::vector<cppflow::tensor>& tf_outputs,
  const std::int32_t nb_pts
);

template void pycomet::model::call(
  const std::vector<double>& inputs,
  std::vector<double>& outputs,
  const std::int32_t nb_pts
);

template std::vector<double> pycomet::model::call(
  const std::vector<double>& inputs,
  const std::int32_t nb_pts
);

template std::vector<std::vector<double>> pycomet::model::call(
  const std::vector<std::vector<double>>& inputs,
  const std::int32_t nb_pts
);
