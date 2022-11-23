#!/bin/sh

TensorFlow_DIR=$CODES_PATH/ML/TensorFlow/v.2.9.1/
export PyCOMET_DIR=$CODES_PATH/ML/PyCOMET/pycomet-1.0-gcc-11.3.0/

export LD_LIBRARY_PATH=$PyCOMET_DIR/lib:$TensorFlow_DIR/lib:${LD_LIBRARY_PATH}
