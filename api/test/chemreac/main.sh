#!/bin/sh

set -e

. ../env.sh

cwd=${PWD}
model_path=${cwd}/model/SavedModel
if [ ! -d "${model_path}" ] 
then
    cd ${model_path}/..
    unzip ${model_path}.zip
    cd ${cwd}
fi

make

./test

rm -rf ${model_path}
make clean
