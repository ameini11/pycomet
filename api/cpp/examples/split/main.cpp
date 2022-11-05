#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <map>

#include "cppflow/cppflow.h"

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

// template<typename T>
// T vector2array(vector<T> const &vec)
// {
//     return *arr = &vec[0];
// }

// template<typename T>
// vector<T> array2vector(T const &arr)
// {
//     return vec(begin(arr), end(arr));
// }

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
    cppflow::tensor t = cppflow::tensor(read_csv(FILEIN_PATH, 0, 0, 3), {300});

    auto shape_t1 = t.shape();
    auto shape = shape_t1.get_data<int64_t>();
    printf("%ld\n", shape[0]);
    printf("%ld\n", shape[1]);

    auto s = cppflow::tensor({100,200});
    auto d = cppflow::tensor(0);
    int64_t n = 1;

    map<std::string, cppflow::tensor> X = {
        {"serving_default_input_1:0", t}, // Branch
        {"serving_default_input_2:0", t}  // Trunk
    };

    printf("%ld\n", n);
    auto t_split = cppflow::split_v(t,s,d,n).get_data<int64_t>()[0];
    printf("%ld\n", n);

    // Write outputs to file
    // write_csv(FILEOUT_PATH, output[0].get_data<float>(), 100, 2);

    return 0;
}


// inline tensor split_v(const tensor& value, const tensor& size_splits, const tensor& split_dim, int64_t num_split, datatype Tlen=static_cast<datatype>(9)) {
