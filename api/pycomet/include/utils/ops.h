#ifndef pycomet_utils_ops_h_
#define pycomet_utils_ops_h_

#include "../includes.h"

namespace pycomet {

  /**
   * @brief Utility functions for vector operations.
   */
  namespace ops {

    /**
     * @brief Print a 2D array.
     *
     * This function prints the elements of a 2D array to the standard output.
     *
     * @tparam T The type of the array elements.
     * @param array The 2D array to be printed.
     */
    template <typename T>
    void print(
      const std::vector<std::vector<T>>& array
    ) {
      for (const std::vector<T>& vector : array) {
        for (const T& value : vector) {
          std::cout << value << ' ';
        }
        std::cout << std::endl;
      }
    }

    /**
     * @brief Print a 1D array.
     *
     * This function prints the elements of a 1D array to the standard output.
     *
     * @tparam T The type of the array elements.
     * @param vector The 1D array to be printed.
     */
    template <typename T>
    void print(
      const std::vector<T>& vector
    ) {
      for (const T& value : vector) {
        std::cout << value << std::endl;
      }
    }

    /**
     * @brief Transpose a 2D array.
     *
     * This function transposes a given 2D array.
     *
     * @tparam T The type of the array elements.
     * @param array The 2D array to be transposed.
     * @return The transposed 2D array.
     */
    template <typename T>
    std::vector<std::vector<T>> transpose(
      const std::vector<std::vector<T>>& array
    ) {
      // Handle empty input array
      if (array.empty() || array[0].empty()) {
        return std::vector<std::vector<T>>();
      }
      // Get dimensions
      const size_t rows = array.size();
      const size_t cols = array[0].size();
      // Perform transposition
      std::vector<std::vector<T>> transposed(cols, std::vector<T>(rows));
      for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
          transposed[j][i] = array[i][j];
        }
      }
      return transposed;
    }

    /**
     * @brief Transpose a flattened 2D array.
     *
     * This function transposes a flattened 2D array with specified dimensions.
     *
     * @tparam T The type of the array elements.
     * @param array The flattened 2D array to be transposed.
     * @param rows The number of rows in the original 2D array.
     * @param cols The number of columns in the original 2D array.
     * @return The transposed flattened 2D array.
     */
    template <typename T>
    std::vector<T> transpose(
      const std::vector<T>& array,
      const std::int32_t rows,
      const std::int32_t cols
    ) {
      // Ensure the size of the input array
      // matches the size defined by 'rows' and 'cols'
      if (array.size() != rows * cols) {
        std::ostringstream message;
        message << "\nFrom pycomet::ops::transpose():"
                << "\n> Input array size doesn't match specified dimensions.";
        throw std::invalid_argument(message.str());
      }
      // Perform transposition
      std::vector<T> transposed(rows * cols);
      for (std::int32_t i = 0; i < rows; i++) {
        for (std::int32_t j = 0; j < cols; j++) {
          transposed[j * rows + i] = array[i * cols + j];
        }
      }
      return transposed;
    }

    /**
     * @brief Flatten a 2D array.
     *
     * This function flattens a given 2D array, optionally
     * transposing it during flattening.
     *
     * @tparam T The type of the array elements.
     * @param array The 2D array to be flattened.
     * @param transpose_in Flag indicating whether to transpose
     *                     the array during flattening.
     * @return The flattened 2D array.
     */
    template <typename T>
    std::vector<T> flatten(
      const std::vector<std::vector<T>>& array,
      const bool transpose_in = false
    ) {
      // Handle empty input array
      if (array.empty() || array[0].empty()) {
        return std::vector<T>();
      }
      // Get dimensions
      const size_t rows = array.size();
      const size_t cols = array[0].size();
      // Perform flattening
      std::vector<T> flattened(rows * cols);
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          if (transpose_in) {
            flattened[j * rows + i] = array[i][j];
          } else {
            flattened[i * cols + j] = array[i][j];
          }
        }
      }
      return flattened;
    }

    /**
     * @brief Reshape a 1D array into a 2D array.
     *
     * This function reshapes a given 1D array into
     * a 2D array with specified dimensions.
     *
     * @tparam T The type of the array elements.
     * @param vector The 1D array to be reshaped.
     * @param shape The target shape of the 2D array (2-element vector).
     * @param transpose_out Flag indicating whether to transpose
     *                      the array after reshaping.
     * @return The reshaped 2D array.
     */
    template <typename T>
    std::vector<std::vector<T>> reshape(
      const std::vector<T>& vector,
      const std::vector<std::int32_t> shape,
      const bool transpose_out = false
    )
    {
      // Error message
      std::ostringstream message;
      message << "From pycomet::ops::reshape(): ";
      // Ensure shape is of size 2
      if (shape.size() != 2) {
        message << "\n> Shape argument must be a 2-element vector.";
        throw std::invalid_argument(message.str());
      }
      // Get dimensions
      const std::int32_t rows = shape[0];
      const std::int32_t cols = shape[1];
      // Ensure the size of the input vector
      // matches the size defined by the shape
      if (rows * cols != vector.size()) {
        message << "\n> Input array size doesn't match specified dimensions.";
        throw std::invalid_argument(message.str());
      }
      // Perform reshaping
      std::vector<std::vector<T>> array(rows, std::vector<T>(cols));
      for (std::int32_t i = 0; i < rows; ++i) {
        for (std::int32_t j = 0; j < cols; ++j) {
          if (transpose_out) {
            array[i][j] = vector[j * rows + i];
          } else {
            array[i][j] = vector[i * cols + j];
          }
        }
      }
      return array;
    }

    /**
     * @brief Invert the major order of a flattened 2D array.
     *
     * This function inverts the major order (row-major to
     * column-major or vice versa) of a given flattened 2D
     * array based on specified dimensions and order preference.
     *
     * @tparam T The type of the array elements.
     * @param x The flattened 2D array to be inverted.
     * @param rows The number of rows in the original 2D array.
     * @param cols The number of columns in the original 2D array.
     * @param col2row Flag indicating the order preference
     *                (true for column-major to row-major).
     */
    template <typename T>
    void invert_major(
      std::vector<T>& x,
      const std::int32_t rows,
      const std::int32_t cols,
      const bool col2row
    ) {
      if ((cols > 1 && col2row) || (rows > 1 && !col2row)) {
        // Transpose the matrix to convert
        // between row-major and column-major
        x = std::move(transpose<T>(x, rows, cols));
      }
    }

  } // namespace ops

} // namespace pycomet

#endif // pycomet_utils_ops_h_
