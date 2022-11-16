module model

  use, intrinsic :: iso_c_binding

  implicit none

  private

  public :: init_model, delete_model, call_model, model_type

  type model_type
    type(c_ptr) :: object = c_null_ptr
    integer(c_int32_t) :: inp_tot_dim = 1
    integer(c_int32_t) :: out_tot_dim = 1
  end type model_type

  interface

    ! Constructor
    function c_init_model(inpfile) result(this) bind(c, name="init_model")
      import
      type(c_ptr) :: this
      character(kind=c_char), dimension(*) :: inpfile
    end function c_init_model

    ! Input/Output dimensions
    function c_get_inp_tot_dim(this) result(inp_tot_dim) bind(c, name="get_inp_tot_dim")
      import
      type(c_ptr), value :: this
      integer(c_int32_t) :: inp_tot_dim
    end function c_get_inp_tot_dim

    function c_get_out_tot_dim(this) result(out_tot_dim) bind(c, name="get_out_tot_dim")
      import
      type(c_ptr), value :: this
      integer(c_int32_t) :: out_tot_dim
    end function c_get_out_tot_dim

    ! Destructor
    subroutine c_delete_model(this) bind(c, name="delete_model")
      import
      type(c_ptr), value :: this
    end subroutine c_delete_model

    ! Evaluate
    function c_call_model(this, inputs, n) result(outputs) bind(c, name="call_model")
      import
      type(c_ptr), value :: this
      type(c_ptr), value :: inputs
      integer(c_int32_t) :: n
      type(c_ptr) :: outputs
    end function c_call_model

  end interface

  contains

  function init_model(inpfile) result(this)
    ! Declare in-out variables
    type(model_type) :: this
    character(*), intent(in) :: inpfile
    ! Initialize TensorFlow model
    this%object = c_init_model(inpfile//c_null_char)
    this%inp_tot_dim = c_get_inp_tot_dim(this%object)
    this%out_tot_dim = c_get_out_tot_dim(this%object)
  end function init_model

  subroutine delete_model(this)
    ! Declare in-out variables
    type(model_type), intent(inout) :: this
    ! Call c++ destructor
    call c_delete_model(this%object)
    ! Re-initialize the `model_type%object` to null pointer
    this%object = c_null_ptr
  end subroutine delete_model

  subroutine call_model(this, inputs, n, f_out_ptr)
    ! Declare in-out variables
    type(model_type), intent(in) :: this
    integer(c_int32_t), intent(in) :: n
    real(c_float), intent(in), target :: inputs(n*this%inp_tot_dim)
    real(c_float), intent(out), dimension(:), pointer :: f_out_ptr
    ! Declare local variables
    type(c_ptr) :: c_out_ptr
    integer :: i
    ! Get c++ outputs pointer with model call function
    c_out_ptr = c_call_model(this%object, c_loc(inputs), n)
    ! Convert c++ pointer to fortran pointer
    call c_f_pointer(c_out_ptr, f_out_ptr, [n*this%out_tot_dim])
  end subroutine call_model

end module model
