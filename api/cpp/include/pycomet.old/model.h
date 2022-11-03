#ifndef PYCOMET_MODEL_H
#define PYCOMET_MODEL_H


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <map>

#include "cppflow/model.h"



namespace pycomet {

    class model
    {
        private:

            // Graph model
            std::shared_ptr<cppflow::model> graph;
            // Inputs/Outputs
            std::map<std::string, std::vector<int64_t>> inputs;
            std::vector<std::string> outputs;

            // Inputs/Outputs general identifier
            const std::string input_id = 'serving_default';
            const std::string output_id = 'StatefulPartitionedCall';

            // Recover identifier-shape pairs for inputs/outputs
            void get_io_info();

        public:

            // Constructor
            model(const std::string& filename);

            // Call functions
            template<T>
            std::vector<std::vector<T>> operator()(std::vector<std::vector<T>> inputs);

            template<T>
            T operator()(T inputs);
    };
}


namespace pycomet {

    model::model(const std::string& filename)
    {
        this->graph = cppflow::model(filename);
        this->graph.get_io_info();
    }

    void model::get_io_info()
    {
        // Get operations names
        std::vector<std::string> ops = this->graph.get_operations();
        // Get operations shapes
        std::vector<std::vector<int64_t>> ops_shape;
        for (auto const &op: ops)
        {
            ops_shape.push_back(this->graph.get_operation_shape(op));
        }
        // Store name-shape pairs
        for (int i = 0; i < ops.size(); i++)
        {
            if (ops[i].find(this->input_id) != std::string::npos)
            {
                this->inputs.emplace(ops[i],ops_shape[i]);
            }
            else if (ops[i].find(this->output_id) != std::string::npos)
            {
                this->outputs.push_back(ops[i]);
            }
        }
    }

    template<T>
    std::vector<std::vector<T>> predict(std::vector<std::vector<T>> inputs)
    {
        // Define inputs
        std::map<std::string, cppflow::tensor> inp;
        for (int i = 0; i < this->inputs.size(); i++)
        {
            auto value = cppflow::tensor(inputs[i], this->inputs[i].second);
            inp.emplace(this->inputs[i].first, value);
        }
        // Perform inference
        auto res = this->graph(inp, this->outputs);
        // Return outputs
        std::vector<std::vector<T>> out;
        for (auto const &r: res)
        {
            out.push_back(r.get_data<T>());
        }
        return out
    }

    template<T>
    T predict(T inputs)
    {
        


        return out
    }

}


#endif


template<T>
T vector2array(std::vector<std::vector<T>> vec)
{
    int i = vec.size()
    int j = vec[0].size()

    std::vector<T> flattened = std::accumulate(
        vec.begin(),
        vec.end(),
        std::vector<T>(),
        [](std::vector<T> &x, std::vector<T> &y)
        {
            x.insert(x.end(), y.begin(), y.end());
            return x;
        }
    );
 
    auto ptr = flattened.data();

}

template<T>
std::vector<std::vector<T>> vec array2vector(T arr)
{
    return {
        std::vector<T>(std::begin(arr[0]),std::end(arr[0])),
        std::vector<T>(std::begin(arr[1]),std::end(arr[1]))
    };
}

template<T>
T vector2array(std::vector<std::vector<T>> vec)
{
    int m = vec.size(), n = vec[0].size();
    T arr[m][n];
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++) {
            arr[i][j] = vec[i][j];
        }
    }
    return arr;
}
