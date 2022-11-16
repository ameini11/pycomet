#include "pycomet.h"


int main() {

  std::string prefix = "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/pycomet/examples/msd/";

  // Load model from file
  auto model = pycomet::model(prefix + "inpfile.json");

  // Read and flatten input data
  auto data = pycomet::csv::read<float>(prefix + "data.csv", {1,101}, {0,3});
  auto vec = pycomet::vec_ops::flatten(data);
  int n = data.size();

  // Perform inference
  auto outputs = model.call(vec, n, true);

  // Write outputs to file
  auto out_arr = pycomet::vec_ops::reshape(outputs, {n, model.out_tot_dim});
  pycomet::csv::write(prefix + "cpp/pred.csv", out_arr);

  return 0;
}
