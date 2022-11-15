#define paste(x)        x
#define append_us(x)    paste(x)_
#define append_y(x,y)   append_us(x)y
#define concat(x, y)    append_y(x,y)

#define read_csv_       concat(read_csv,t)
#define write_csv_      concat(write_csv,t)


subroutine read_csv_(filename, row_lim, col_lim, delimiter, data)
  ! Declarations
  ! ---------------------------------------------------------------------------
  ! Declare in-out variables
  character(*), intent(in) :: filename
  integer, intent(in), optional, dimension(2) :: row_lim
  integer, intent(in), optional, dimension(2) :: col_lim
  character(16), intent(in), optional :: delimiter
  tt, intent(out), allocatable, dimension(:,:) :: data
  ! Declare local variables
  integer, parameter :: huge = int(1.d5)
  integer :: i, j, c, r, n, m, rc, fu
  character(len=huge) :: line
  tt, allocatable, dimension(:) :: row_data
  ! Optional arguments
  integer, dimension(2) :: row_lim_ = (/1,huge/)
  integer, dimension(2) :: col_lim_ = (/1,huge/)
  character(1) :: delimiter_ = ','
  ! Optional arguments
  ! ---------------------------------------------------------------------------
  if (present(row_lim)) row_lim_ = row_lim
  if (present(col_lim)) col_lim_ = col_lim
  if (present(delimiter)) delimiter_ = delimiter
  ! File handling
  ! ---------------------------------------------------------------------------
  open(                                                                       &
    action='read',                                                            &
    file=filename,                                                            &
    newunit=fu,                                                               &
    access='sequential',                                                      &
    form='formatted',                                                         &
    status='old'                                                              &
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
  ! Allocate variables
  ! ---------------------------------------------------------------------------
  allocate(data(                                                              &
    min(n, row_lim_(2)-row_lim_(1)),                                          &
    min(m, col_lim_(2)-col_lim_(1))                                           &
  ))
  allocate(row_data(m))
  ! Read values
  ! ---------------------------------------------------------------------------
  r = 1
  do i = 1,n
    c = 1
    if ((row_lim_(1) <= i).and.(i < row_lim_(2))) then
      read(fu, *) row_data
      do j = 1,m
        if ((col_lim_(1) <= j).and.(j < col_lim_(2))) then
          data(r,c) = row_data(j)
          c = c + 1
        endif
      enddo
      r = r + 1
    else
      read(fu, *) line
    endif
  enddo
  close(fu)
end subroutine read_csv_

subroutine write_csv_(filename, n, data)
  ! Declarations
  ! ---------------------------------------------------------------------------
  ! Declare in-out variables
  character(*), intent(in) :: filename
  integer, intent(in) :: n
  tt, intent(in), dimension(:,:) :: data
  ! Declare local variables
  integer :: i, fu
  ! Core function
  ! ---------------------------------------------------------------------------
  open(                                                                       &
    action='write',                                                           &
    file=filename,                                                            &
    newunit=fu                                                                &
  )
  do i = 1,n
    write(fu, *)  data(i, :)
  enddo
  close(fu)
end subroutine write_csv_


#undef t
#undef tt

#undef paste
#undef append_us
#undef append_y
#undef concat

#undef read_csv_
#undef write_csv_