module model

	implicit none

	public : init_model, call_model, delete_model, alias_outputs

	private
		real*8, pointer :: outputs(:,:)
	contains

	subroutine init_model(file)
		character(*), allocatable :: file
		call c_init_model(file)
	end subroutine init_model

	subroutine delete_model()
		call c_delete_model()
	end subroutine delete_model

	function call_model(inputs) result(p)
		real*8, allocatable :: inputs(:,:)
		real*8, pointer :: p(:,:)
		call c_call_model(inputs)
		p => outputs(1:nx,1:ny)
	end function call_model

	subroutine alias_outputs(out_ptr)
		real*8, allocatable, target :: & out_ptr(:,:)

		outputs => out(1:nx,1:ny)

	end subroutine alias_outputs

end module model



! crectangle_module.f90
module model

	use, intrinsic :: iso_c_binding, only: c_double, c_char, c_ptr, c_null_ptr
	implicit none

	private

	type model_type
		private
		type(c_ptr) :: object = c_null_ptr
	end type model_type

	interface

		function c_init_model(file) result(this) bind(c, name="init_model")
			import
			type(c_ptr) :: this
	  		character(*) :: file
		end function c_init_model

		subroutine c_delete_model(this) bind(c, name="delete_model")
			import
			type(c_ptr), value :: this
		end subroutine c_delete_model

		function c_call_model(this, inputs) result(outputs) bind(c, name="call_model")
			import
			real(c_double) :: outputs(:,:)
			real(c_double) :: inputs(:,:)
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

	function init_model(this, file)
		type(model_type), intent(out) :: this
		character(*), allocatable :: file
		this%object = c_init_model(file)
	end function init_model

	subroutine delete_model(this)
		type(model_type), intent(inout) :: this
		call c_delete_model(this%object)
		this%object = c_null_ptr
	end subroutine delete_model

	function call_model(this, inputs) result(outputs)
		type(model_type), intent(in) :: this
		real(8), intent(in), allocatable :: inputs(:,:)
		real(8), allocatable :: outputs(:,:)
		outputs = c_call_model(this%object, inputs)
	end function call_model

end module model


! crectangle_module.f90
module CRectangle_module
  use, intrinsic :: ISO_C_Binding, only: C_int, C_ptr, C_NULL_ptr
  implicit none
  private
  type CRectangle_type
    private
    type(C_ptr) :: object = C_NULL_ptr
  end type CRectangle_type
  interface
    function C_CRectangle__new (a, b) result(this) bind(C,name="CRectangle__new")
      import
      type(C_ptr) :: this
      integer(C_int), value :: a, b
    end function C_CRectangle__new
    subroutine C_CRectangle__delete (this) bind(C,name="CRectangle__delete")
      import
      type(C_ptr), value :: this
    end subroutine C_CRectangle__delete
    function C_CRectangle__area (this) result(area) bind(C,name="CRectangle__area")
      import
      integer(C_int) :: area
      type(C_ptr), value :: this
    end function C_CRectangle__area
  end interface
  interface new
    module procedure CRectangle__new
  end interface new
  interface delete
    module procedure CRectangle__delete
  end interface delete
  interface area
    module procedure CRectangle__area
  end interface area
  public :: new, delete, area, CRectangle_type
contains
! Fortran wrapper routines to interface C wrappers
  subroutine CRectangle__new(this,a,b)
    type(CRectangle_type), intent(out) :: this
    integer :: a,b
    this%object = C_CRectangle__new(int(a,C_int),int(b,C_int))
  end subroutine CRectangle__new
  subroutine CRectangle__delete(this)
    type(CRectangle_type), intent(inout) :: this
    call C_CRectangle__delete(this%object)
    this%object = C_NULL_ptr
  end subroutine CRectangle__delete
  function CRectangle__area(this) result(area)
    type(CRectangle_type), intent(in) :: this
    integer :: area
    area = C_CRectangle__area(this%object)
  end function CRectangle__area
end module CRectangle_module


! program main
!   use CRectangle_module
!   type(CRectangle_type) :: rect
!   call new(rect,3,4)
!   write(*,*) 'rect area: ',area(rect)
!   call delete(rect)
! end program main