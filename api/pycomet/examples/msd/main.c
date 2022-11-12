#include "pycomet.h"


std::vector<double> read_csv(std::string filename, int skiprows, int col_s, int col_e)
{
    std::ifstream file(filename);
    std::vector<std::vector<double>> array;

    if (file)
    {
        int row = 0;
        std::string line;

        while (std::getline(file, line))
        {
            if (row > skiprows)
            {
                std::stringstream sep(line);
                std::string field;

                array.push_back(std::vector<double>());

                int col = 0;
                while (std::getline(sep, field, ','))
                {
                    if (col_s <= col && col_e > col)
                    {
                        array.back().push_back(std::stod(field));
                    }
                    col++;
                }
            }
            row++;
        }
    }

    return flatten(array);
}

void write_csv(std::string filename, std::vector<double> vec, int r, int c){

    ofstream file(filename);

    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            file << vec[i*c+j];
            if (j != c - 1)
            {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

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
