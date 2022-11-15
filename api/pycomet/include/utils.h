#ifndef pycomet_utils_h_
#define pycomet_utils_h_


#include <limits>
#include "includes.h"


namespace pycomet
{
  namespace vec_ops
  {
    // Transpose a 2D array ---------------------------------------------------
    template <typename T>
    std::vector<std::vector<T>> transpose(
      const std::vector<std::vector<T>> &vec
    )
    {
      std::vector<std::vector<T>> vec_trans(vec[0].size(), std::vector<T>());
      for (int i = 0; i < vec.size(); i++)
      {
        for (int j = 0; j < vec[i].size(); j++)
        {
          vec_trans[j].push_back(vec[i][j]);
        };
      };
      return vec_trans;
    }

    // Flatten a 2D array into 1D array ---------------------------------------
    template <typename T>
    std::vector<T> flatten(
      const std::vector<std::vector<T>> &vec2D,
      const bool transpose_in = false
    )
    {
      std::vector<T> vec1D;
      auto _vec2D = (transpose_in == true) ? transpose(vec2D) : vec2D;
      for (auto const &v: _vec2D)
        vec1D.insert(vec1D.end(), v.begin(), v.end());
      return vec1D;
    }

    // Reshape a 1D array into 2D array ---------------------------------------
    template <typename T>
    std::vector<std::vector<T>> reshape(
      const std::vector<T> &vec1D,
      const std::vector<int> shape,
      const bool transpose_out = false
    )
    {
      std::vector<std::vector<T>> vec2D;
      int start = 0;
      int end = 0;
      for (int i = 0; i < shape[0]; ++i)
      {
        end = start + shape[1];
        std::vector<T> vec(vec1D.begin() + start, vec1D.begin() + end);
        start = end;
        vec2D.push_back(vec);
      };
      auto _vec2D = (transpose_out == true) ? transpose(vec2D) : vec2D;
      return _vec2D;
    }
  }

  namespace csv
  {
    template <typename T>
    std::vector<std::vector<T>> read(
      const std::string filename,
      const std::vector<int> row_lim = {-1,std::numeric_limits<int>::max()},
      const std::vector<int> col_lim = {-1,std::numeric_limits<int>::max()}
    )
    {
      std::ifstream file(filename);
      std::vector<std::vector<T>> array;
      if (file)
      {
        int row = 0;
        std::string line;
        while (std::getline(file, line))
        {
          if (row_lim[0] <= row && row < row_lim[1])
          {
            std::stringstream sep(line);
            std::string field;
            array.push_back(std::vector<T>());
            int col = 0;
            while (std::getline(sep, field, ','))
            {
              if (col_lim[0] <= col && col < col_lim[1])
              {
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
      const std::vector<T> &vec,
      const std::vector<int> shape,
      const bool rowmajor = true,
      const int precision = 6
    )
    {
      std::ofstream file(filename);
      for (int i = 0; i < shape[0]; ++i)
      {
        for (int j = 0; j < shape[1]; ++j)
        {
          auto v = (rowmajor == true) ? vec[i*shape[1]+j] : vec[j*shape[0]+i];
          file << std::setprecision(precision) << v;
          if (j != shape[1]-1)
          {
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
      const std::vector<std::vector<T>> &vec,
      const int precision = 6
    )
    {
      std::ofstream file(filename);
      for (int i = 0; i < vec.size(); i++)
      {
        for (int j = 0; j < vec[i].size(); j++)
        {
          file << std::setprecision(precision) << vec[i][j];
          if (j != vec[i].size()-1)
          {
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