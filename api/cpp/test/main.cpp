#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cppflow/ops.h"
#include "cppflow/model.h"

using namespace std;


template<typename T>
vector<T> flatten(vector<vector<T>> const &vec)
{
    vector<T> flattened;
    for (auto const &v: vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

vector<float> read_csv(string filename, int skiprows, int col_s, int col_e)
{
    ifstream file(filename);
    vector<vector<float>> array;

    if (file)
    {
        int row = 0;
        string line;

        while (getline(file, line))
        {
            if (row > skiprows)
            {
                stringstream sep(line);
                string field;

                array.push_back(vector<float>());

                int col = 0;
                while (getline(sep, field, ','))
                {
                    if (col_s <= col && col_e > col)
                    {
                        array.back().push_back(stod(field));
                    }
                    col++;
                }
            }
            row++;
        }
    }

    return flatten(array);
}

void write_csv(string filename, vector<float> vec, int r, int c){

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

    std::cout << "input_2: " << input_2 << std::endl;

    // Load the model from file
    auto model = cppflow::model(std::string(MODEL_PATH));
    
    // Perform inference
    auto output = model(
        {
            {"serving_default_input_1:0", input_1}, // Branch
            {"serving_default_input_2:0", input_2}  // Trunk
        },
        {"StatefulPartitionedCall:0"}
    );

    // Write outputs to file
    write_csv(FILEOUT_PATH, output[0].get_data<float>(), 100, 2);

    return 0;
}
