#!/bin/bash -i
set -e

# Loading the PyCOMET module
module load pycomet/1.0-gcc-9.3.0-cuda-11.2.0-tf-2.10.0-gpu

# Set proper environmental variables
export TF_CPP_MIN_LOG_LEVEL=0
export TF_ENABLE_ONEDNN_OPTS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export CUDA_VISIBLE_DEVICES=1
