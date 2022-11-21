#include "pycomet.h"


int main() {

  std::string prefix = "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/pycomet/examples/msd/";
  int n = 100;

  // Load model from file
  auto model = pycomet::model(prefix + "cpp/inpfile.json");

  // Read and flatten input data
  std::vector<float> inp_vec;
  int d = 0;
  for (std::uint16_t i = 0; i < model.inputs_dim.size(); ++i)
    {
      // Extract i-th input
      auto d_i = model.inputs_dim[i];
      auto inp = pycomet::csv::read<float>(
        prefix + "data.csv", {1, n+1}, {d, d+d_i}
      );
      d += d_i;
      // Append the i-th input
      auto inp_flt = pycomet::vec_ops::flatten(inp);
      // Append inputs
      inp_vec.insert(inp_vec.end(), inp_flt.begin(), inp_flt.end());
    }

  // Perform inference
  auto out_vec = model.call(inp_vec, n);

  // Reorder and reshape outputs
  std::vector<float> outputs;
  int start = 0;
  int end = 0;
  int delta;
  for (std::uint16_t i = 0; i < model.outputs_dim.size(); ++i)
  {
    // Extract i-th output
    delta = model.outputs_dim[i] * n;
    end = start + delta;
    std::vector<float> y(out_vec.begin() + start, out_vec.begin() + end);
    start = end;
    // Append output
    std::vector<float> y_ = pycomet::vec_ops::invert_major(
      y, n, model.outputs_dim[i], true
    );
    outputs.insert(outputs.end(), y_.begin(), y_.end());
  }

  // Write outputs to file
  auto out_arr = pycomet::vec_ops::reshape(
    outputs, {model.out_tot_dim, n}, true
  );
  pycomet::csv::write(prefix + "cpp/pred.csv", out_arr);

  return 0;
}
