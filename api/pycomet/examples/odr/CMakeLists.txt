cmake_minimum_required(VERSION 3.10)
project(odr)
set(CMAKE_CXX_STANDARD 17)

find_library(pycomet_LIB libpycomet.so)
message(STATUS ${pycomet_LIB})

add_executable(odr main.cpp)
target_link_libraries(odr PRIVATE ${pycomet_LIB})
