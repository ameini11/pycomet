program main

	use, intrinsic :: iso_c_binding
	use model

    ! implicit none

	character(10) :: filename = '1000000000'

	integer(c_int8_t) :: i, j
	integer(c_int64_t) :: n = 10
	real(8), allocatable, dimension(:,:) :: inp
	real(8), allocatable, dimension(:,:) :: out
	type(model_type) :: mdl


	mdl = init_model(trim(filename))

	allocate(inp(n,mdl%inp_tot_dim))

	do i=1,n
		do j = 1,mdl%inp_tot_dim
			inp(i, j) = (i)**2
		end do
		write(*,*) inp(i, :)
	end do


	out = call_model(mdl, inp, n)


	do i=1,n
		write(*,*) out(i, :)
	end do


	call delete_model(mdl)

end program main
