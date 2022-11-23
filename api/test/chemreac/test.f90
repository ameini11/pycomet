program test

  use pcm_model

  implicit none

  ! Declarations
  character(64) :: cwd
  character(32) :: inpfile = "/model/inpfile.json"
  type(model_type) :: model
  
  call getcwd(cwd)

  ! Load model from file
  call init_model(trim(cwd) // trim(inpfile), model)

  ! Delete model
  call delete_model(model)

end program test
