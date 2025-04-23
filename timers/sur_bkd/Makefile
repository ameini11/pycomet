CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -I${TF2_INC} -I${TF_INC} -MMD -MP -ffast-math
LDFLAGS = -L${TF2_LIB} -ltf2 -L${TF_LIB} -ltensorflow
PROGRAM = main.x
SRCS = main.cpp
OBJECTS = $(SRCS:.cpp=.o)
DEPENDS = $(SRCS:.cpp=.d)

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $<

-include $(DEPENDS)

.PHONY: clean

clean:
	rm -f *.o *.out *.d $(PROGRAM) *~
