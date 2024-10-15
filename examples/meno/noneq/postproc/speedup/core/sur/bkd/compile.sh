#!/bin/bash -i
set -e

compile() {
  make clean
  make
  mv main.x main_${device}.x
  make clean
}

devices=("cpu" "gpu")

for device in ${devices[*]}; do
  if [ ${device} == "gpu" ]; then
    module load pycomet/1.0-gcc-9.3.0-cuda-11.2.0-tf-2.10.0-gpu
  else
    module load pycomet/1.0-gcc-11.3.0-tf-2.10.0-cpu
  fi
  compile
  module purge
done
