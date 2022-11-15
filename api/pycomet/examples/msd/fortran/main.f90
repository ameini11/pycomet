program main

    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env
    use model
    use utils

    implicit none

    character(64) :: prefix = "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/pycomet/examples/msd/"
    character(16) :: inpfile = "inpfile.json"
    character(16) :: datafile = "data.csv"
    character(16) :: predfile = "fortran/pred.csv"
    real(kind=real32), allocatable, dimension(:,:) :: data
    real(kind=real32), allocatable, dimension(:) :: vec
    integer, dimension(2) :: row_lim = (/2,102/)
    integer, dimension(2) :: col_lim = (/1,4/)
    integer :: i
    type(model_type) :: mdl

  ! auto model = trim(prefix) // "inpfile.json"

  !   zz = trim(xx) // trim(yy)

  !   integer(c_int8_t) :: i, j
  !   integer(c_int64_t) :: n = 10
  !   real(8), allocatable, dimension(:,:) :: inp
  !   real(8), allocatable, dimension(:,:) :: out
  
    call read_csv_float(                                                      &
      trim(prefix) // trim(datafile), row_lim, col_lim, data=data             &
    )

    mdl = init_model(trim(prefix) // trim(inpfile))

    ! allocate(inp(n,mdl%inp_tot_dim))

    vec = reshape(data, (/100*3/))    ! prints "2 2"

    do i=1,100*3
        write(*,*) vec(i)
    end do
    ! write(*,*) vec(:)

    call write_csv_float(                                                     &
      trim(prefix) // trim(predfile), 100, data=data                          &
    )

    ! out = call_model(mdl, inp, n)


    ! do i=1,n
    !     write(*,*) out(i, :)
    ! end do


    call delete_model(mdl)

end program main
