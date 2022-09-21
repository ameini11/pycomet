cmake_minimum_required(VERSION 3.10)

project(cpp)

find_package(cppflow REQUIRED)

target_include_directories(
  cpp PUBLIC
  cppflow::cppflow
)

target_link_libraries(
  cpp PUBLIC
  cppflow::cppflow
)
