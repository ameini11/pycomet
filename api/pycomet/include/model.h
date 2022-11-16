#ifndef pycomet_model_h
#define pycomet_model_h


#include "includes.h"
#include <cppflow/cppflow.h>


namespace pycomet
{
  // Model class
  // ==========================================================================
  class model
  {
    private:

      // TensorFlow model
      std::unique_ptr<cppflow::model> tfmodel;
      std::string path_to_model;
      std::vector<std::uint8_t> *config = nullptr;

      // Inputs/Outputs
      std::vector<std::string> inputs_id;
      std::vector<std::string> outputs_id;
      std::vector<std::int32_t> inputs_dim;
      std::vector<std::int32_t> outputs_dim;
      std::vector<std::int32_t> check_io_ops(
        const std::vector<std::string> identifiers
      );
      const std::string inputs_id_prefix = "serving_default";
      const std::string outputs_id_prefix = "StatefulPartitionedCall";

      // IO operations
      std::vector<std::tuple<std::string, std::vector<int64_t>>> ops;

      void get_ops_info();

      // Input parser
      void parse_inputs(const std::string inpfile);

      // Set global TF context
      void set_global_context(std::vector<std::uint8_t> *config);

    public:

      // Constructor
      explicit model(const std::string inpfile);

      // Destructor
      ~model() = default;

      // Inputs/Outputs total dimensions
      std::int32_t inp_tot_dim;
      std::int32_t out_tot_dim;

      // Call functions
      template <typename T>
      std::vector<T> call(
        const std::vector<T> inputs,
        const std::int32_t n,
        const bool rowmajor = false
      );

      template <typename T>
      std::vector<std::vector<T>> call(
        const std::vector<std::vector<T>> inputs,
        const std::int32_t n,
        const bool rowmajor = false,
        const bool transpose_in = false,
        const bool transpose_out = false
      );

      template <typename T>
      std::vector<T> call(
        const T *inputs,
        const std::int32_t n,
        const bool rowmajor = false
      );
  };
}


#endif // pycomet_model_h
