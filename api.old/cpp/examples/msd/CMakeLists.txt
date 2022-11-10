cmake_minimum_required(VERSION 3.10)
project(msd)

add_executable(msd main.cpp)
target_link_libraries(msd pycomet)
target_compile_definitions(msd PUBLIC
	MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/SavedModel"
	FILEIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/inputs.csv"
	FILEOUT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/outputs.csv"
)
