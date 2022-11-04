# C++ directives
CC=g++
CFLAGS=-lstdc++

# Fortran directives
FC=gfortran

all: clean
	$(CC) -c model_wrapper.cpp
	$(FC) -c model.F90
	$(FC) main.F90 model.o model_wrapper.o -o main $(CFLAGS)

clean:
	rm -rf *.mod *.o main
