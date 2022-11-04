module model

	use, intrinsic :: iso_c_binding, only: c_double, c_char, c_ptr, c_null_ptr

	implicit none

	private

	type model_type
		private
		type(c_ptr) :: object = c_null_ptr
	end type model_type

	interface

		function c_init_model(filename) result(this) bind(c, name="init_model")
			import
			type(c_ptr) :: this
			character(c_char) :: filename
		end function c_init_model

		subroutine c_delete_model(this) bind(c, name="delete_model")
			import
			type(c_ptr), value :: this
		end subroutine c_delete_model

		function c_call_model(this, inputs) result(outputs) bind(c, name="call_model")
			import
			real(c_double) :: outputs
			real(c_double) :: inputs
			type(c_ptr), value :: this
		end function c_call_model

	end interface

	interface init
		module procedure init_model
	end interface init

	interface delete
		module procedure delete_model
	end interface delete

	interface call
		module procedure call_model
	end interface call

	public :: init, delete, call, model_type

	contains

	function init_model(filename) result(this)
		type(model_type) :: this
		character(c_char), intent(in) :: filename
		this%object = c_init_model(filename)
	end function init_model

	subroutine delete_model(this)
		type(model_type), intent(inout) :: this
		call c_delete_model(this%object)
		this%object = c_null_ptr
	end subroutine delete_model

	function call_model(this, inputs) result(outputs)
		type(model_type), intent(in) :: this
		real(c_double), intent(in) :: inputs
		real(c_double) :: outputs
		write(*,*) 'F90:call_model inputs: ', inputs
		outputs = c_call_model(this%object, inputs)
	end function call_model

end module model


! Example
! -------------------------------------
program main

	use, intrinsic :: iso_c_binding, only: c_double, c_char
	use model

    implicit none

	character(c_char) :: filename = '1'

	real(c_double) :: inp = 1
	real(c_double) :: out
	type(model_type) :: mdl

	write(*,*) 'F90 inp: ', inp

	mdl = init(filename)

	out = call(mdl, inp)
	write(*,*) 'F90 out: ', out

	call delete(mdl)

end program main
