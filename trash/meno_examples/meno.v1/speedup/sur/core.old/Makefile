CXX = g++
CXXFLAGS = -g -pg -std=c++17 -Wall -I${PYCOMET_INC} -I${TF_INC}
LDFLAGS = -L${PYCOMET_LIB} -lpycomet -L${TF_LIB} -ltensorflow
PROGRAM = main.x
SRCS = main.cpp
OBJECTS = $(SRCS:.cpp=.o)

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $<

.PHONY: clean

clean:
	rm -f *.o *.out *~ $(PROGRAM)
