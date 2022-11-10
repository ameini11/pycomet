subroutine grid bridge(nx, ny, nz, a)
	use model, only: alias_outputs
	integer, intent(in) :: nx,ny,nz
	real*8, intent(in) :: a(nx*ny*nz)
	call alias_outputs(nx, ny, nz, a)
end subroutine grid bridg


g++ -c model_wrapper.c && gfortran -c model.F90 && gfortran -o model model_wrapper.o model.o -lstdc++