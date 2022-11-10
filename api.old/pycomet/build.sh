CC=g++
CFLAGS="-DDEBUG -Wall -g -c -std=c++17"
INCLUDE="-I/home/zanardi/Codes/ML/TensorFlow/v.2.9.1/include -I/home/zanardi/Codes/ML/CppFlow/cppflow/include -I/home/zanardi/APPLICATIONS/JSON/json/include"
LDFLAGS="-L/home/zanardi/Codes/ML/TensorFlow/v.2.9.1/lib"
LDLIBS="-lstdc++ -ltensorflow"
SOURCES="pycomet.c model.c interface.c main.c"
# OBJECTS=$(SOURCES:.c=.o)



$CC $CFLAGS $INCLUDE $LDFLAGS $LDLIBS $SOURCES
# g++ $CFLAGS $INCLUDE model.c -o model.o
# g++ model.o $CFLAGS $INCLUDE interface.c -o interface.o
# g++ $CFLAGS $INCLUDE main.c -o main.o
$CC *.o $LDFLAGS $LDLIBS -o main