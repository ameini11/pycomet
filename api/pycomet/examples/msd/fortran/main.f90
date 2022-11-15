program main

    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env
    use model
    use utils

    implicit none

    character(64) :: prefix = "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/pycomet/examples/msd/"
    character(16) :: inpfile = "inpfile.json"
    character(16) :: datafile = "data.csv"
    real(kind=real32), allocatable, dimension(:,:) :: data
    integer, dimension(2) :: row_lim = (2,102)
    integer, dimension(2) :: col_lim = (1,4)
    type(model_type) :: mdl

  ! auto model = trim(prefix) // "inpfile.json"

  !   zz = trim(xx) // trim(yy)

  !   integer(c_int8_t) :: i, j
  !   integer(c_int64_t) :: n = 10
  !   real(8), allocatable, dimension(:,:) :: inp
  !   real(8), allocatable, dimension(:,:) :: out
  
    data = read_csv(trim(prefix) // trim(datafile), row_lim, col_lim)

    mdl = init_model(trim(prefix) // trim(inpfile))

    ! allocate(inp(n,mdl%inp_tot_dim))

    ! do i=1,n
    !     do j = 1,mdl%inp_tot_dim
    !         inp(i, j) = (i)**2
    !     end do
    !     write(*,*) inp(i, :)
    ! end do


    ! out = call_model(mdl, inp, n)


    ! do i=1,n
    !     write(*,*) out(i, :)
    ! end do


    call delete_model(mdl)

end program main
