program main

	! use, intrinsic :: iso_c_binding, only: c_double, c_char
	use model

    implicit none

	character(10) :: filename = '1000000000'

	real(8) :: inp = 0.1
	real(8) :: out
	type(model_type) :: mdl

	mdl = init_model(trim(filename))

	out = call_model(mdl, inp)

	call delete_model(mdl)

end program main
