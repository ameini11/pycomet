module model

	use, intrinsic :: iso_c_binding

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
			character(kind=c_char), dimension(*) :: filename
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

	public :: init_model, delete_model, call_model, model_type

	contains

	function init_model(filename) result(this)
		type(model_type) :: this
    	character(*), intent(in) :: filename
		this%object = c_init_model(filename//c_null_char) 
		write(*,*) 'f90::constr >> model constructed from file ', filename(:)
	end function init_model

	subroutine delete_model(this)
		type(model_type), intent(inout) :: this
		call c_delete_model(this%object)
		this%object = c_null_ptr
		write(*,*) 'f90::destr >> model destructed.'
	end subroutine delete_model

	function call_model(this, inputs) result(outputs)
		type(model_type), intent(in) :: this
		real(c_double), intent(in) :: inputs
		real(c_double) :: outputs
		outputs = c_call_model(this%object, inputs)
		write(*,*) 'f90::call >> inputs = ', inputs
	end function call_model

end module model
