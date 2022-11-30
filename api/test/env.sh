#!/bin/bash

# Two options:

# a) Exporting installing directories for PyCOMET and TensorFlow
# -----------------------------------------------------------------------------
# export TENSORFLOW_DIR=$CODES_PATH/ML/TensorFlow/v.2.9.1/
# export PYCOMET_DIR=$CODES_PATH/ML/PyCOMET/pycomet/api/install
# export LD_LIBRARY_PATH=$PYCOMET_DIR/lib:$TENSORFLOW_DIR/lib:${LD_LIBRARY_PATH}

# b) Loading the PyCOMET module
# -----------------------------------------------------------------------------
module load pycomet
