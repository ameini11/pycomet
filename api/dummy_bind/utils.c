#include <iostream>
#include <vector>
#include "utils.h"


// Flatten a 2D array into 1D array
template <typename T>
std::vector<T> flatten(
    const std::vector<std::vector<T>> &vec
)
{
    std::vector<T> result;
    for (auto const &v: vec)
        result.insert(result.end(), v.begin(), v.end());
    return result;
};

// Transpose a 2D array
template <typename T>
std::vector<std::vector<T>> transpose(
    const std::vector<std::vector<T>> &vec
)
{
    if (vec.size() == 0)
        return;
    std::vector<std::vector<T>> vec_trans(vec[0].size(), std::vector<T>());
    for (int i = 0; i < vec.size(); i++)
    {
        for (int j = 0; j < vec[i].size(); j++)
        {
            vec_trans[j].push_back(vec[i][j]);
        };
    };
    return vec_trans;
};

// Reshape a 1D array into 2D array
template <typename T>
std::vector<std::vector<T>> reshape(
    const std::vector<T> &vec,
    const std::int64_t x_dim,
    const std::int64_t y_dim
)
{
    std::vector<std::vector<T>> array;
    int start = 0;
    int end = 0;
    for (int j = 0; j < y_dim; ++j)
    {
        end = start + x_dim;
        std::vector<T> column(vec.begin() + start, vec.begin() + end);
        start = end;
        array.push_back(column);
    };
    std::vector<std::vector<T>> array_trans = transpose(array)
    return array_trans;
};
