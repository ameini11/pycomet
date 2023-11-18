#ifndef pycomet_utils_h_
#define pycomet_utils_h_


#include "includes.h"
#include <cppflow/cppflow.h>


namespace pycomet {

  /**
   * @brief Namespace containing custom data structures for PyCOMET.
   *
   * The data namespace defines custom data structures used in
   * the PyCOMET library. Two primary structures are defined:
   *
   * 1. list_tuple_tensor: A vector of tuples, where each tuple consists
   *    of a string identifier and a corresponding TensorFlow tensor.
   *
   * 2. list_tuple_vector: A vector of tuples, where each tuple consists
   *    of a string identifier and a vector of a specified type T.
   *
   * These structures are designed to facilitate the manipulation
   * and organization of data within the PyCOMET library.
   */
  namespace data {

    /**
     * @brief Represents a vector of tuples, each containing a string
     * identifier and a corresponding TensorFlow tensor.
     */
    using list_tuple_tensor = std::vector<std::tuple<std::string, cppflow::tensor>>;

    /**
     * @brief Represents a vector of tuples, each containing a string
     * identifier and a vector of a specified type T.
     *
     * This structure is often used for storing input/output data
     * associated with PyCOMET operations.
     *
     * @tparam T The type of data stored in the vector.
     */
    template <typename T>
    using list_tuple_vector = std::vector<std::tuple<std::string, std::vector<T>>>;

  } // namespace data

  // Vector operations
  // ==================================
  namespace ops
  {
    // Printing
    // --------------------------------
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

    template <typename T>
    void print(
      const std::vector<T>& vector
    ) {
      for (const T& value : vector) {
        std::cout << value << std::endl;
      }
    }

    // Transpose a 2D array
    // --------------------------------
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

    // Transpose a flattened 2D array
    // --------------------------------
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

    // Flatten a 2D array
    // --------------------------------
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

    // Reshape a 1D array into 2D array
    // --------------------------------
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

    // Change the major order of a flattened 2D array
    // --------------------------------
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
  }

  // CSV Files
  // ==================================
  namespace csv
  {
    template <typename T>
    std::vector<std::vector<T>> read(
      const std::string filename,
      const std::vector<int> row_lim = {0,std::numeric_limits<int>::max()},
      const std::vector<int> col_lim = {0,std::numeric_limits<int>::max()}
    )
    {
      std::ifstream file(filename);
      std::vector<std::vector<T>> array;
      if (file) {
        int row = 0;
        std::string line;
        while (std::getline(file, line)) {
          if (row_lim[0] <= row && row < row_lim[1]) {
            std::stringstream sep(line);
            std::string field;
            array.push_back(std::vector<T>());
            int col = 0;
            while (std::getline(sep, field, ',')) {
              if (col_lim[0] <= col && col < col_lim[1]) {
                array.back().push_back(std::stod(field));
              }
              col++;
            }
          }
          row++;
        }
      }
      return array;
    }

    template <typename T>
    void write(
      const std::string filename,
      const std::vector<T>& vec,
      const std::vector<int> shape,
      const bool rowmajor = true,
      const int precision = 6
    ) {
      std::ofstream file(filename);
      for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
          auto v = (rowmajor == true) ? vec[i*shape[1]+j] : vec[j*shape[0]+i];
          file << std::setprecision(precision) << v;
          if (j != shape[1]-1) {
            file << ",";
          }
        }
        file << "\n";
      }
      file.close();
    }

    template <typename T>
    void write(
      const std::string filename,
      const std::vector<std::vector<T>>& vec,
      const int precision = 6
    ) {
      std::ofstream file(filename);
      for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[i].size(); j++) {
          file << std::setprecision(precision) << vec[i][j];
          if (j != vec[i].size()-1) {
            file << ",";
          }
        }
        file << "\n";
      }
      file.close();
    }
  }

}


#endif // pycomet_utils_h_