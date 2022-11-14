#include "pycomet.h"


int main() {

  std::string prefix = "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/pycomet/examples/msd/";

  // Load the model from file
  auto model = pycomet::model(prefix + "inpfile.json");

  // Read and flatten input data
  auto data = pycomet::csv::read<float>(prefix + "data.csv", {1,101}, {0,3});
  auto n = data.size();
  auto d = data[0].size();
  auto vec = pycomet::vec_ops::flatten(data, true);
  // for (auto const &i: vec)
  //   std::cout << "\n " << i << std::endl;

  // Perform inference
  auto outputs = model.call(vec, n);
  // auto outputs = model.call(inpvec, static_cast<std::int32_t>(n));
  for (auto const &i: outputs)
    std::cout << "\n " << i << std::endl;

  return 0;
}
