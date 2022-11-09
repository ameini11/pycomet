#include <iostream>
#include <cstdint>
#include <vector>

#include "utils.h"


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

// Flatten a 2D array into 1D array
template <typename T>
std::vector<T> flatten(
    const std::vector<std::vector<T>> &vec2D,
    const bool transpose_in
)
{
    std::vector<T> vec1D;
    auto _vec2D = (transpose_in == true) ? transpose(vec2D) : vec2D;
    for (auto const &v: _vec2D)
        vec1D.insert(vec1D.end(), v.begin(), v.end());
    return vec1D;
};

// Reshape a 1D array into 2D array
template <typename T>
std::vector<std::vector<T>> reshape(
    const std::vector<T> &vec1D,
    const std::vector<std::int64_t> shape,
    const bool transpose_out
)
{
    std::vector<std::vector<T>> vec2D;
    int start = 0;
    int end = 0;
    for (int j = 0; j < shape[1]; ++j)
    {
        end = start + shape[0];
        std::vector<T> vec(vec1D.begin() + start, vec1D.begin() + end);
        start = end;
        vec2D.push_back(vec);
    };
    auto _vec2D = (transpose_out == true) ? transpose(vec2D) : vec2D;
    return _vec2D;
};
