# C++ directives
CC=g++
CFLAGS=-lstdc++

# Fortran directives
FC=gfortran

all: clean
	$(CC) -DDEBUG -g -c model_c.c
	$(FC) -DDEBUG -g -c model_f.f90
	$(FC) -DDEBUG -g main.f90 model_c.o model_f.o -o main $(CFLAGS)

clean:
	rm -rf *.mod *.o main
