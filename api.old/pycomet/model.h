#ifndef pycomet_model_h
#define pycomet_model_h


#include "includes.h"
#include "cppflow/cppflow.h"
#include "vec_ops.h"


namespace pycomet
{
	// Model class
	// ========================================================================
	class model {

		private:

		    // TensorFlow model
		    std::unique_ptr<cppflow::model> tfmodel;
		    std::string path_to_model;

		    // Inputs/Outputs
		    std::vector<std::string> inp_id;
		    std::vector<std::string> out_id;
		    std::vector<std::int32_t> inp_dim;
		    std::vector<std::int32_t> out_dim;
			std::vector<std::int32_t> check_io_ops(
				const std::vector<std::string> identifiers
			);

		    // IO operations
		    std::vector<std::tuple<std::string, std::vector<int64_t>>> ops;

		    void get_ops_info();

		    // Input parser
		    void parse_inputs(const std::string inpfile);

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
				std::vector<T> inputs,
				const std::int32_t n
			);

		    template <typename T>
			std::vector<std::vector<T>> call(
				std::vector<std::vector<T>> inputs,
				const std::int32_t n,
			    const bool transpose_in = false,
			    const bool transpose_out = false
			);

			template <typename T>
			T *call(
				T *inputs,
				const std::int32_t n
			);
	};
}


#endif // pycomet_model_h
