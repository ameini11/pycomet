# # C++ directives
# CC=g++
# CFLAGS=-lstdc++

# LDFLAGS = -lstdc++ -L/home/zanardi/Codes//ML/TensorFlow/v.2.9.1/lib/ -ltensorflow

# all: clean
# 	$(CC) -DDEBUG -g -c model.c
# 	$(CC) -DDEBUG -g main.c model.o -o main $(CFLAGS)

# clean:
# 	rm -rf *.mod *.o main

CC = g++
CFLAGS = -DDEBUG -Wall -g -c -std=c++17
INCLUDE = -I/home/zanardi/Codes/ML/TensorFlow/v.2.9.1/include -I/home/zanardi/Codes/ML/CppFlow/cppflow/include -I/home/zanardi/APPLICATIONS/JSON/json/include
LDFLAGS = -L/home/zanardi/Codes/ML/TensorFlow/v.2.9.1/lib
LDLIBS = -lstdc++ -ltensorflow
SOURCES = pycomet.c model.c interface.c main.c
OBJECTS = $(SOURCES:.c=.o)

EXE = main

all: $(SOURCES) $(EXE)

$(EXE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@ 

.c.o:
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@

clean:
	rm -rf *.so *.o *.a $(EXE)


# all: $(EXE)

# $(EXE): $(SOURCES)
# 	$(CC) $(CFLAGS) $(SOURCES) $(INCLUDE) $(LDFLAGS) $(LDLIBS) -o $@

# clean:
# 	rm -rf *.so *.o *.a $(EXE)


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zanardi/Codes/ML/TensorFlow/v.2.9.1/lib