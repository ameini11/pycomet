program execution_time

    use omp_lib
    
    implicit none
    real(8) :: start_time, end_time, elapsed_time
    integer, parameter :: num_iterations = 5 ! Number of iterations for testing
    integer :: iter
    integer, parameter :: N = 10            ! Size of the square matrices

    real, dimension(N,N) :: A, B, C

    ! Start measuring time
    start_time = omp_get_wtime()

    ! Your program or code block to be timed goes here
    call matrix_multiplication(num_iterations, N, A, B, C)
    do iter = 1,2
        write(*,*) "YEPPP", iter
    end do
    
    ! End measuring time
    end_time = omp_get_wtime()
    
    ! Calculate elapsed time
    elapsed_time = end_time - start_time
    
    ! Output the result
    print *, "Elapsed time:", elapsed_time, "seconds"
    
contains

    subroutine matrix_multiplication(iterations, size, A, B, C)
        implicit none
        integer, intent(in) :: iterations, size
        real, dimension(size,size), intent(inout) :: A, B, C
        integer :: iter, i, j, k
        
        do iter = 1, iterations
            ! Initialize matrices A and B
            A = 1.0
            B = 2.0

            ! Perform matrix multiplication C = A * B
            do i = 1, size
                do j = 1, size
                    C(i,j) = 0.0
                    do k = 1, size
                        C(i,j) = C(i,j) + A(i,k) * B(k,j)
                    end do
                end do
            end do
        end do
    end subroutine matrix_multiplication

end program execution_time
