#ifndef pycomet_model_h
#define pycomet_model_h


#include "includes.h"
#include <cppflow/cppflow.h>


namespace pycomet
{
  class model
  {
    private:

      // TensorFlow model
      std::string path_to_model;
      std::unique_ptr<cppflow::model> tfmodel;
      std::vector<std::uint8_t> *config = nullptr;

      // Inputs/Outputs
      std::vector<std::int32_t> check_io_ops(
        const std::vector<std::string> identifiers
      );
      const std::string inputs_id_prefix = "serving_default";
      const std::string outputs_id_prefix = "StatefulPartitionedCall";

      // Evaluating parameters
      std::int32_t batch_size = -1;
      bool rowmajor = true;

      // IO operations
      std::vector<std::tuple<std::string, std::vector<int64_t>>> ops;

      void get_ops_info();

      // Input parser
      void parse_inputs(const std::string inpfile);

      // Set global TF context
      void set_global_context(std::vector<std::uint8_t> *config);

      // Inputs/Outputs manipulations
      template <typename T>
      std::vector<T> compose_outputs(
        std::vector<cppflow::tensor> outputs,
        const std::int32_t n
      );
      
      template <typename T>
      std::vector<std::vector<T>> compose_inputs(
        std::vector<T> inputs,
        const std::int32_t n
      );

      template <typename T>
      std::vector<std::tuple<std::string, cppflow::tensor>> convert_inputs(
        std::vector<std::vector<T>> inputs,
        const std::int32_t n
      );

    public:

      // Constructor
      explicit model(const std::string inpfile);

      // Destructor
      ~model() = default;

      // Inputs/Outputs
      std::vector<std::string> inputs_id;
      std::vector<std::string> outputs_id;
      std::vector<std::int32_t> inputs_dim;
      std::vector<std::int32_t> outputs_dim;

      // Inputs/Outputs total dimensions
      std::int32_t inp_tot_dim;
      std::int32_t out_tot_dim;

      // Call functions
      template <typename T>
      std::vector<T> call(
        std::vector<T> inputs,
        const std::int32_t n
      );

      template <typename T>
      std::vector<std::vector<T>> call(
        std::vector<std::vector<T>> inputs,
        const std::int32_t n
      );

      // template <typename T>
      // std::vector<T> call(
      //   const T *inputs,
      //   const std::int32_t n,
      //   const bool rowmajor = false
      // );
  };
}


#endif // pycomet_model_h
