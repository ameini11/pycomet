#!/bin/bash

# Python Library installation
if [[ -d "./dist" ]]
then
    pip3 uninstall dist/*.whl
    rm -rf dist build prode.egg-info
fi
pipreqs --force ./
python3 setup.py bdist_wheel
pip3 install dist/*.whl

