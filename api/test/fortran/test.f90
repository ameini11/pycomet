program test

  use pcm_model

  implicit none

  ! Declarations
  character(64) :: prefix = "/home/zanardi/Codes/ML/PyCOMET/pycomet/api/test/chemreac/model/"
  character(32) :: inpfile = "inpfile.json"
  type(model_type) :: model

  ! Load model from file
  call init_model(trim(prefix) // trim(inpfile), model)

  ! Delete model
  call delete_model(model)

end program test
