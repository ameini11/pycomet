#include "cppflow/ops.h"
#include "cppflow/model.h"

#include "model.h"


model::model(const std::string& filename) {

    this->graph = cppflow::model(filename);

    // Get operations names
    std::string input_id = 'serving_default';
    std::string output_id = 'StatefulPartitionedCall';

    std::vector<std::string> ops = this->graph.get_operations();
    for (auto const &op: ops) {
        if (op.find(input_id) != string::npos) {
            this->inputs_sign.insert(this->inputs_sign.end(), op)
        } else if (op.find(output_id) != string::npos) {
            this->outputs_sign.insert(this->outputs_sign.end(), op)
        }
    }

    // Get operations shapes
    std::vector<std::string> ops = this->graph.get_operations();
    for (auto const &op: ops) {
        if (op.find(input_id) != string::npos) {
            this->inputs_sign.insert(this->inputs_sign.end(), op)
        } else if (op.find(output_id) != string::npos) {
            this->outputs_sign.insert(this->outputs_sign.end(), op)
        }
    }
}
