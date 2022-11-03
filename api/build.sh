#!/bin/sh -i
set -e


# CppFlow
SRC_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/cpp/
BUILD_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/build
INSTALL_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/install

# TensorFlow
TensorFlow_DIR=$CODES_PATH/ML/TensorFlow/v.2.9.1/

# CppFlow
CppFlow_DIR=$CODES_PATH/ML/CppFlow/install/

# Install
rm -rf $BUILD_DIR $INSTALL_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

cmake $SRC_DIR \
	-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_EXAMPLES=ON \
	-Dcppflow_INCLUDE_DIRS=$CppFlow_DIR/include \
	-Dcppflow_LIBRARIES=$CppFlow_DIR/lib/ \
	-Dtensorflow_INCLUDE_DIRS=$TensorFlow_DIR/include \
	-Dtensorflow_LIBRARIES=$TensorFlow_DIR/lib/libtensorflow.so

make -j 4
make install
