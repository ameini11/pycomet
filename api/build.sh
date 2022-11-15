#!/bin/sh
set -e


# PyCOMET
SRC_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/pycomet/
BUILD_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/build
INSTALL_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/install

# TensorFlow
TensorFlow_DIR=$CODES_PATH/ML/TensorFlow/v.2.9.1/
# CppFlow
CppFlow_DIR=$CODES_PATH/ML/CppFlow/cppflow/
# Json
Json_DIR=$APPLICATIONS_PATH/JSON/json/

# Install
rm -rf $BUILD_DIR $INSTALL_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

cmake $SRC_DIR \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DBUILD_EXAMPLES=ON \
  -Djson_INCLUDE_DIR=$Json_DIR/include \
  -Dcppflow_INCLUDE_DIR=$CppFlow_DIR/include \
  -Dtensorflow_INCLUDE_DIR=$TensorFlow_DIR/include \
  -Dtensorflow_LIBRARY=$TensorFlow_DIR/lib/libtensorflow.so

make -j 4
make install
