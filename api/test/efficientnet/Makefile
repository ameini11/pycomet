CXX = g++
CXXFLAGS = -g -pg -std=c++17 -Wall -I${CPPFLOW_INC} -I${TF_INC}
LDFLAGS = -L${TF_LIB} -ltensorflow
PROGRAM = test
SRCS = test.cpp
OBJECTS = $(SRCS:.cpp=.o)

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $<

.PHONY: clean

clean:
	rm -f *.o *.out *~ $(PROGRAM)
