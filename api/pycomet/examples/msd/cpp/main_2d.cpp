#include "pycomet.h"


int main() {

  std::string prefix = "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/pycomet/examples/msd/";
  int n = 100;

  // Load model from file
  auto model = pycomet::model(prefix + "cpp/inpfile.json");

  // Read and flatten input data
  auto data = pycomet::csv::read<float>(
    prefix + "data.csv", {1, n+1}, {0, model.inp_tot_dim}
  );

  // Perform inference
  auto outputs = model.call(data, n);

  // Write outputs to file
  pycomet::csv::write(prefix + "cpp/pred.csv", outputs);

  return 0;
}
