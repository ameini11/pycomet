cmake_minimum_required(VERSION 3.10)
project(test)

add_executable(test main.cpp)
target_link_libraries(test pycomet)
target_compile_definitions(test PUBLIC
	FILEIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/inputs.csv"
	FILEOUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/outputs.csv"
)
