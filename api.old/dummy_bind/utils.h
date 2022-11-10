#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdint>
#include <vector>


// Flatten a 2D array into 1D array
template <typename T>
std::vector<T> flatten(
    const std::vector<std::vector<T>> &vec
);

// Transpose a 2D array
template <typename T>
std::vector<std::vector<T>> transpose(
    const std::vector<std::vector<T>> &vec2D,
    const bool transpose_in = true
);

// Reshape a 1D array into 2D array
template <typename T>
std::vector<std::vector<T>> reshape(
    const std::vector<T> &vec1D,
    const std::vector<std::int64_t> shape,
    const bool transpose_out = true
);

#endif // UTILS_H