#!/bin/sh

TensorFlow_DIR=$CODES_PATH/ML/TensorFlow/v.2.9.1/
export PyCOMET_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/install

export LD_LIBRARY_PATH=$PyCOMET_DIR/lib:$TensorFlow_DIR/lib:${LD_LIBRARY_PATH}