#!/bin/sh -i
set -e


# PyCOMET
SRC_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/
BUILD_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api_build
INSTALL_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api_install

# TensorFlow
TF_VER=2.9.1
TF_DIR=$CODES_PATH/ML/TensorFlow/v.$TF_VER

# CppFlow
CF_DIR=$CODES_PATH/ML/CppFlow/install

# Install
rm -rf $BUILD_DIR $INSTALL_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

echo $CF_DIR

cmake \
	-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
	-DCMAKE_PREFIX_PATH=$TF_DIR \
	-DCMAKE_MODULE_PATH=$CF_DIR \
	$SRC_DIR

make -j
make install