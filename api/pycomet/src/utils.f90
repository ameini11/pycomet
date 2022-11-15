module utils

  use, intrinsic :: iso_fortran_env

  implicit none

  private :: columns_count

  public :: read_csv !, write_csv

  ! interface

  !   function c_read_csv(filename, row_lim, col_lim) result(data) bind(c, name="read_csv")
  !     import
  !     character(kind=c_char), dimension(*) :: filename
  !     integer(c_int), optional, dimension(2) :: row_lim
  !     integer(c_int), optional, dimension(2) :: col_lim
  !     type(c_ptr) :: data
  !   end function c_read_csv

  !   subroutine c_write_csv(filename, shape, data) bind(c, name="write_csv")
  !     import
  !     character(kind=c_char), dimension(*) :: filename
  !     integer(c_int), dimension(2) :: shape
  !     type(c_ptr) :: data
  !   end subroutine c_write_csv

  ! end interface

  ! interface read_csv
  !   module procedure :: read_csv_int, read_csv_float, read_csv_double
  ! end interface
  
  ! interface write_csv
  !   module procedure :: write_csv_int, write_csv_float, write_csv_double
  ! end interface

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

  function read_csv(filename, row_lim, col_lim, delimiter) result(data)
    ! Declare in-out variables
    character(*), intent(in) :: filename
    integer, intent(in), optional, dimension(2) :: row_lim
    integer, intent(in), optional, dimension(2) :: col_lim
    character(16), intent(in), optional :: delimiter
    real(kind=real32), allocatable, dimension(:,:) :: data
    ! Declare local variables
    integer, parameter :: huge = int(1.d6)
    integer :: i, j, c, r, n, m, rc, fu
    character(len=huge) :: line
    real, allocatable, dimension(:) :: row_data
    ! Optional arguments
    integer, dimension(2) :: row_lim_ = (1,huge)
    integer, dimension(2) :: col_lim_ = (1,huge)
    character(16) :: delimiter_ = ','

    ! Optional arguments
    if (present(row_lim)) row_lim_ = row_lim
    if (present(col_lim)) col_lim_ = col_lim
    if (present(delimiter)) delimiter_ = delimiter
    ! Open file
    open(                   &
      action='read',        &
      file=filename,        &
      newunit=fu,           &
      access='sequential',  &
      form='formatted',     &
      status='old'          &
    )
    ! Count lines 
    n = 0
    m = 0
    do
      read(fu, '(A)', iostat=rc) line
      if ((m == 0).and.(n > row_lim_(1))) then
        m = columns_count(trim(line), trim(delimiter_))
      endif
      if (rc /= 0) exit
      n = n + 1
    end do
    ! Rewind file
    rewind(fu)
    ! Allocate variable
    allocate(data(                      &
      min(n, row_lim_(2)-row_lim_(1)),  &
      min(m, col_lim_(2)-col_lim_(1))   &
    ))
    allocate(row_data(m))
    ! Read values
    r = 1
    do i = 1,n
      read(fu, *) row_data
      write(*,*) row_data
      c = 1
      if ((row_lim_(1) <= i).and.(i < row_lim_(2))) then
        do j = 1,m
          if ((col_lim_(1) <= j).and.(j < col_lim_(2))) then
            data(r,c) = row_data(j)
            c = c + 1
          endif
        enddo
        r = r + 1
      endif
    enddo
  end function read_csv

  ! #define t int
  ! #define tt integer(c_int)
  ! #include "utils_t.f90"

  ! #define t float
  ! #define tt real(c_float)
  ! #include "utils_t.f90"

  ! #define t double
  ! #define tt real(c_double)
  ! #include "utils_t.f90"

end module utils
