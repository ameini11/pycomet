#include "pycomet.h"



int main() {

    // Read inputs from file
    auto input_1 = cppflow::tensor(read_csv(FILEIN_PATH, 0, 0, 2), {100, 2});
    auto input_2 = cppflow::tensor(read_csv(FILEIN_PATH, 0, 2, 3), {100, 1});

    // Load the model from file
    auto model = cppflow::model(std::std::string(MODEL_PATH));

    std::std::vector<std::tuple<std::std::string, cppflow::tensor>> X = {
        {"serving_default_input_1:0", input_1}, // Branch
        {"serving_default_input_2:0", input_2}  // Trunk
    };

    // Perform inference
    auto output = model(X, {"StatefulPartitionedCall:0"});

    // auto output = model(
    //     {
    //         {"serving_default_input_1:0", input_1}, // Branch
    //         {"serving_default_input_2:0", input_2}  // Trunk
    //     },
    //     {"StatefulPartitionedCall:0"}
    // );

    // Write outputs to file
    write_csv(FILEOUT_PATH, output[0].get_data<double>(), 100, 2);

    return 0;
}
