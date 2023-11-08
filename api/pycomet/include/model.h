#ifndef pycomet_model_h
#define pycomet_model_h


#include "includes.h"
#include "utils.h"
#include <cppflow/cppflow.h>


namespace pycomet
{
  class model
  {
    private:

      // TensorFlow model
      std::string path_to_model;
      std::unique_ptr<cppflow::model> tfmodel;
      std::vector<std::string>* config = nullptr;

      // Inputs/Outputs
      std::vector<std::int32_t> check_io_ops(
        const std::vector<std::string> identifiers
      );
      const std::string inputs_id_prefix = "serving_default";
      const std::string outputs_id_prefix = "StatefulPartitionedCall";
      const std::int32_t max_nb_io = 100;

      // Data major ordering
      bool rowmajor = true;

      // Batched inference
      std::int32_t batch_size = -1;

      // IO operations
      pycomet::data::list_tuple_vector<int64_t> ops;

      void get_ops_info();

      // Input parser
      void parse_inputs(const std::string inpfile);

      // Set global TF context
      void set_global_context(const std::vector<std::string>& config);

      // Inputs/Outputs manipulations
      template <typename T>
      pycomet::data::list_tuple_tensor compose_inputs(
        const std::vector<T>& inputs,
        const std::int32_t nb_pts
      );

      template <typename T>
      void compose_outputs(
        std::vector<T>& outputs,
        const std::vector<cppflow::tensor>& tf_outputs,
        const std::int32_t nb_pts
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
      void evaluate(
        const std::vector<T>& inputs,
        std::vector<T>& outputs,
        const std::int32_t nb_pts
      );

      template <typename T>
      void call(
        const std::vector<T>& inputs,
        std::vector<T>& outputs,
        const std::int32_t nb_pts
      );

      template <typename T>
      std::vector<T> call(
        const std::vector<T>& inputs,
        const std::int32_t nb_pts
      );

      template <typename T>
      std::vector<std::vector<T>> call(
        const std::vector<std::vector<T>>& inputs,
        const std::int32_t nb_pts
      );

  };
}


#endif // pycomet_model_h
