# C++ directives
CC=g++
CFLAGS=-lstdc++

# Fortran directives
FC=gfortran

all: clean
	$(CC) -DDEBUG -g -c model_wrapper.c
	$(FC) -DDEBUG -g -c model.F90
	$(FC) -DDEBUG -g main.F90 model.o model_wrapper.o -o main $(CFLAGS)

clean:
	rm -rf *.mod *.o main
