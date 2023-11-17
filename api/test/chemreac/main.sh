#!/bin/bash
set -e


source ./../env.sh

text="> Unpacking TF model ..."
print_text "\${text}" "="
cwd=${PWD}
model_path=${cwd}/model/SavedModel
if [ ! -d "${model_path}" ]; then
  cd ${cwd}/model/
  unzip ${model_path}.zip
  cd ${cwd}
fi

text="> Compiling ..."
print_text "\${text}" "="
load_pycomet true
make clean
make

text="> Running ..."
print_text "\${text}" "="
./test

text="> Cleaning ..."
print_text "\${text}" "="
make clean
rm -rf ${model_path}
