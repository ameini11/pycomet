cmake_minimum_required(VERSION 3.10)
project(odr)

add_executable(odr main.cpp)
target_link_libraries(odr pycomet)