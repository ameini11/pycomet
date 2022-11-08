#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>


// Flatten a 2D array into 1D array
template <typename T>
std::vector<T> flatten(
    const std::vector<std::vector<T>> &vec
);

// Transpose a 2D array
template <typename T>
std::vector<std::vector<T>> transpose(
    const std::vector<std::vector<T>> &vec
);

// Reshape a 1D array into 2D array
template <typename T>
std::vector<std::vector<T>> reshape(
    const std::vector<T> &vec,
    const std::int64_t x_dim,
    const std::int64_t y_dim
);

#endif // UTILS_H