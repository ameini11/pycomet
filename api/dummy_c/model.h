#ifndef model_h
#define model_h


#include "includes.h"


// Model class
/* ------------------------------------------------------------------------- */
class model {

	private:

	    // TensorFlow model
	    cppflow::model tf_model;

	    // Inputs/Outputs
		const std::vector<std::string> inp_id;
		const std::vector<std::string> out_id;
		const std::vector<std::int32_t> inp_dim;
		const std::vector<std::int32_t> out_dim;
		std::vector<std::int32_t> check_io_ops(
			const std::vector<std::string> identifiers
		);

	    // IO operations
	    const std::vector<std::tuple<std::string, std::vector<int32_t>>> ops;

	    void get_ops_info();

	    // Input parser
	    void parse_inputs(const std::string inpfile);

	public:

	    // Constructor
	    explicit model(const std::string inpfile);

	    // Destructor
		~model() = default;

	    // Inputs/Outputs total dimensions
		const std::int32_t inp_tot_dim;
		const std::int32_t out_tot_dim;

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
}


// External bindings
/* ------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C"
{
#endif

	model *init_model(char *inpfile);

	std::int32_t get_inp_tot_dim(model *mdl);

	std::int32_t get_out_tot_dim(model *mdl);

	void delete_model(model *mdl);

	double *call_model(model *mdl, double *inputs, std::int32_t *n);

#ifdef __cplusplus
}
#endif


#endif // model_h
