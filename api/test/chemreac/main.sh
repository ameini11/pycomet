#!/bin/bash

set -e

. ../env.sh

cwd=${PWD}
model_path=${cwd}/model/SavedModel
if [ ! -d "${model_path}" ] 
then
    cd ${cwd}/model/
    unzip ${model_path}.zip
    cd ${cwd}
fi

make clean
make

./test

rm -rf ${model_path}
make clean
