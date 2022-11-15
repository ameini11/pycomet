#define concat(x, y)    x_y

#define read_csv        concat(read_csv,t)
#define write_csv       concat(write_csv,t)

function read_csv(filename, row_lim, col_lim) result(f_data_ptr)
  ! Declare in-out variables
  character(*), intent(in) :: filename
  integer(c_int), intent(in), optional, dimension(2) :: row_lim
  integer(c_int), intent(in), optional, dimension(2) :: col_lim
  tt, dimension(:), pointer :: f_data_ptr
  ! Declare local variables
  type(c_ptr) :: c_data_ptr
  ! Get c++ outputs pointer with model call function
  c_data_ptr = c_read_csv(filename, row_lim, col_lim)
  ! Convert c++ pointer to fortran pointer
  call c_f_pointer(c_out_ptr, f_out_ptr, [n*this%inp_tot_dim])


  type(model_type) :: this
  character(*), intent(in) :: inpfile
  ! Initialize TensorFlow model
  this%object = c_init_model(filename//c_null_char)
  this%inp_tot_dim = c_get_inp_tot_dim(this%object)
  this%out_tot_dim = c_get_out_tot_dim(this%object)
end function read_csv

subroutine write_csv(filename, shape, data)
  ! Declare in-out variables
  type(model_type) :: this
  character(*), intent(in) :: inpfile
  ! Initialize TensorFlow model
  this%object = c_init_model(inpfile//c_null_char)
  this%inp_tot_dim = c_get_inp_tot_dim(this%object)
  this%out_tot_dim = c_get_out_tot_dim(this%object)
end subroutine write_csv

#undef t
#undef tt

#undef read_csv
#undef write_csv



    function c_read_csv(filename, row_lim, col_lim) result(data) bind(c, name="read_csv")
      import
      character(kind=c_char), dimension(*) :: filename
      integer(c_int), optional, dimension(2) :: row_lim
      integer(c_int), optional, dimension(2) :: col_lim
      type(c_ptr) :: data
    end function c_read_csv

    subroutine c_write_csv(filename, shape, data) bind(c, name="write_csv")
      import
      character(kind=c_char), dimension(*) :: filename
      integer(c_int), dimension(2) :: shape
      type(c_ptr) :: data
    end subroutine c_write_csv



    use, intrinsic :: iso_fortran_env, only: error_unit

    implicit none

    character(len=*), parameter :: filename = 'data.txt'

    character(len=72) :: street
    integer           :: i, file_unit, price, rc

    open(action='read', file=filename, iostat=rc, newunit=fileunit)

    if (rc /= 0) then
      write(error_unit, '(3a, i0)') 'Reading file "', filename, '" failed: ', rc
      stop
    end if

    do
      read(fileunit, *, iostat=rc) i, street, price
      if (rc /= 0) exit
    end do

    close (file_unit)




    ! Declarations
    character*100 :: input_path,input_filename_ast, output_path, output_filename_ast
    character*118 :: str_line
    character*30, dimension (4) :: str_output    !for header of output file for Matlab
    integer :: i,iflag, n_lines
    type(NEO), dimension (:), allocatable :: asteroids   
    real(pr), dimension (:,:), allocatable :: elem_matrix
    
     ! Definition of constants, paths names and file names
   
    ! input_path = "\OneDrive\MSc_Thesis\Projects\NEOs_orbits\InputFiles\"
   
    
    ! Reading of asteroids data

    open(action='read', file=filename, newunit=fileunit)

    ! open(unit = iu_in, file = trim(input_path) // trim(input_filename_ast), status='old', & 
    !      access = 'sequential',form = 'formatted', action='read')
    
    ! Count lines 
    n = 0 
    do
      read(fileunit,*,iostat=io)
      if (n == row_lim(1))
      endif
      if (io/=0) exit
      n = n + 1
    end do

        mdl = model%init_model(trim(prefix) // trim(file))


    ! Rewind file
    rewind(fileunit)
    
    ! Allocate variable
    allocate(data(n,m))
    
    read (iu_in,'(a35)') str_line
        do i = 1,55
            read(iu_in,*) asteroids(i)         
        enddo
    
    close(unit = iu_in, status='keep') 

    !allocate(elem_matrix(55,3))
    !
    !do i =1,55
    !          elem_matrix = (asteroids%a(i), asteroids%e(i), asteroids%i(i))
    !enddo
    
    print *, asteroids
    
    ! Convert Mean anomaly to true anomaly


        integer :: N, i, lines_in_file
    real*8, allocatable :: cs(:,:)

    N = lines_in_file('datafile.txt') ! a function I wrote, which works correctly

    allocate(cs(N,2))

    open(15, 'datafile.txt', status='old')
    read(15,*) cs

    do i=1,N
        print *, cs(i,1), cs(i,2)
    enddo
    
    
    
    stop