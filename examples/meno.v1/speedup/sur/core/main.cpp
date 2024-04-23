#include <cmath>
#include <chrono>
#include <nlohmann/json.hpp>
#include "pycomet/pycomet.h"

using json = nlohmann::json;


// Statistics
// ====================================
double mean(const std::vector<double>);

double stdev(const std::vector<double>);

double mean(const std::vector<double> samples)
{
  double sum = 0.0;
  int size = samples.size();
  for (int i = 0; i < size; i++) {
    sum += samples[i];
  }
  return sum / size;
}

double stdev(const std::vector<double> samples)
{
  double var = 0.0;
  double mu = mean(samples);
  int size = samples.size();
  for (int i = 0; i < size; i++) {
    var += pow(samples[i] - mu, 2);
  }
  return sqrt(var/size);
}


// Main
// ====================================
int main() {

  std::cout << "Started!" << std::endl;

  // Global inputs
  // > Initialize JSON object reader
  std::ifstream file("./inputs.json");
  json inputs = json::parse(file);
  // > Read inputs
  std::string modelfile = inputs["path_to_modelfile"];
  std::string datafile = inputs["path_to_datafile"];
  int n = inputs["nb_points"];
  int m = inputs["nb_repeat"];

  // Load model from file
  std::cout << "> Loading model" << std::endl;
  auto model = pycomet::model(modelfile);

  // Read and flatten input data
  std::cout << "> Loading input data" << std::endl;
  std::vector<double> x;
  int d = 0;
  for (std::uint16_t i = 0; i < model.inputs_dim.size(); ++i) {
    // > Extract i-th input
    auto d_i = model.inputs_dim[i];
    auto x_i = pycomet::csv::read<double>(datafile, {1, n+1}, {d, d+d_i});
    d += d_i;
    // > Append i-th input
    auto x_i_flt = pycomet::ops::flatten(x_i);
    x.insert(x.end(), x_i_flt.begin(), x_i_flt.end());
  }

  // Allocate output tensor
  std::vector<double> y(n * model->out_tot_dim);
  // > Test evaluation
  y = model.call(x, n);

  if (m > 0) {

    // Perform inference
    std::cout << "> Evaluating inference time" << std::endl;
    // > Loop over m evaluations
    std::vector<double> timings;
    for (int i = 0; i < m; ++i) {
      auto s = std::chrono::system_clock::now();
      y = model.call(x, n);
      auto e = std::chrono::system_clock::now();
      double delta = std::chrono::duration<double>(e-s).count();
      timings.push_back(delta/n);
    }
    // Statistics
    std::vector<double> stats;
    stats.push_back(mean(timings));
    stats.push_back(stdev(timings));
    auto stats_2d = pycomet::ops::reshape(stats, {1,2});
    pycomet::csv::write("./stats.csv", stats_2d);

  } else {

    // Write outputs to file
    std::cout << "> Saving predictions" << std::endl;
    auto out_arr = pycomet::ops::reshape(y, {n, model.out_tot_dim}, false);
    pycomet::csv::write("./pred.csv", out_arr);

  }

  std::cout << "Done!" << std::endl;

  return 0;
}
