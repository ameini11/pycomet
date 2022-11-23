FCOMP = gfortran
FCFLAGS = -g -O2
LDFLAGS = -I${PyCOMET_DIR}/mod -L${PyCOMET_DIR}/lib -lpycomet
PROGRAM =  test 
SRCS = test.f90
OBJECTS = $(SRCS:.f90=.o)

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(FCOMP) $(FCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.f90
	$(FCOMP) $(FCFLAGS) -c $< $(LDFLAGS)

.PHONY: clean

clean:
	rm -f *.o *.mod *.MOD *~ $(PROGRAM)
