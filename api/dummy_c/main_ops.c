#include "includes.h"
#include "utils.h"

// using namespace std;


// template<typename T>
// vector<T> flatten(vector<vector<T>> const &vec)
// {
//     vector<T> flattened;
//     for (auto const &v: vec) {
//         flattened.insert(flattened.end(), v.begin(), v.end());
//     }
//     return flattened;
// }

// // template<typename T>
// // T vector2array(vector<T> const &vec)
// // {
// //     return *arr = &vec[0];
// // }

// // template<typename T>
// // vector<T> array2vector(T const &arr)
// // {
// //     return vec(begin(arr), end(arr));
// // }

// vector<float> read_csv(string filename, int skiprows, int col_s, int col_e)
// {
//     ifstream file(filename);
//     vector<vector<float>> array;

//     if (file)
//     {
//         int row = 0;
//         string line;

//         while (getline(file, line))
//         {
//             if (row > skiprows)
//             {
//                 stringstream sep(line);
//                 string field;

//                 array.push_back(vector<float>());

//                 int col = 0;
//                 while (getline(sep, field, ','))
//                 {
//                     if (col_s <= col && col_e > col)
//                     {
//                         array.back().push_back(stod(field));
//                     }
//                     col++;
//                 }
//             }
//             row++;
//         }
//     }

//     return flatten(array);
// }

// void write_csv(string filename, vector<float> vec, int r, int c){

//     ofstream file(filename);

//     for (int i = 0; i < r; ++i)
//     {
//         for (int j = 0; j < c; ++j)
//         {
//             file << vec[i*c+j];
//             if (j != c - 1)
//             {
//                 file << ",";
//             }
//         }
//         file << "\n";
//     }

//     file.close();
// }



int main() {

    std::vector<int> v = {0,1,2,3,4,5,6,7,8,9};

    std::vector<std::int64_t> shape = {2,5};

    // std::vector<std::vector<int>> a1 = reshape(v, shape, true);
    // std::cout << "a1 : " << std::endl;
    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < 5; i++)
    //     {
    //         std::cout << a1[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::vector<std::vector<int>> a2 = reshape(v, shape, false);
    std::cout << "a2 : " << std::endl;
    for (int i = 0; i < shape[0]; i++)
    {
        for (int j = 0; j < shape[1]; j++)
        {
            std::cout << a2[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // std::vector<int> v1 = flatten(a1, true);
    // std::cout << "v1 : " << std::endl;
    // for (int i = 0; i < 10; i++)
    //     std::cout << v1[i] << std::endl;

    std::vector<int> v2 = flatten(a2, true);
    std::cout << "v2 : " << std::endl;
    for (int i = 0; i < 10; i++)
        std::cout << v2[i] << std::endl;

    return 0;
}
