program main

  use, intrinsic :: iso_c_binding
  use pcm_model
  use utils

  implicit none

  ! Declarations
  character(64) :: prefix = "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/pycomet/examples/msd/"
  character(16) :: inpfile = "inpfile.json"
  character(16) :: datafile = "data.csv"
  character(16) :: predfile = "fortran/pred.csv"
  real(c_float), allocatable, dimension(:,:) :: data
  real(c_float), allocatable, dimension(:) :: inputs, outputs
  integer, dimension(2) :: row_lim = (/2,102/)
  integer, dimension(2) :: col_lim = (/1,4/)
  integer(c_int32_t) :: n
  type(model_type) :: mdl

  ! Load model from file
  mdl = init_model(trim(prefix) // trim(inpfile))

  ! Read input data
  call read_csv_float(                                                        &
    trim(prefix) // trim(datafile), row_lim, col_lim, data=data               &
  )

  ! Flatten input data
  n = row_lim(2)-row_lim(1)
  allocate(inputs(size(data)))
  inputs = reshape(data, (/size(data)/))

  ! Perform inference
  allocate(outputs(200))
  call call_model(mdl, n, inputs, outputs)
  data = reshape(outputs, (/n, mdl%out_tot_dim/))

  ! Write outputs to file
  call write_csv_float(                                                       &
    trim(prefix) // trim(predfile), n, data=data                              &
  )

  ! Delete model
  call delete_model(mdl)

end program main
