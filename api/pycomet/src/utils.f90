module utils

  use, intrinsic :: iso_c_binding

  implicit none

  private :: columns_count

  public :: read_csv, write_csv

  interface read_csv
    module procedure :: read_csv_int, read_csv_float, read_csv_double
  end interface
  
  interface write_csv
    module procedure :: write_csv_int, write_csv_float, write_csv_double
  end interface

  contains

  recursive function columns_count(line, delimiter) result(count)
    ! Declare in-out variables
    character(*), intent(in) :: line
    character(*), intent(in) :: delimiter
    integer :: delimiter_idx, count
    ! Function core
    delimiter_idx = index(line, delimiter)
    if (delimiter_idx .eq. 0) then
      count = 1
    else
      count = 1 + columns_count(line(delimiter_idx+1:), delimiter)
    end if
  end function columns_count

#define t int
#define tt integer(c_int32_t)
#include "utils_t.f90"

#define t float
#define tt real(c_float)
#include "utils_t.f90"

#define t double
#define tt real(c_double)
#include "utils_t.f90"

end module utils
