#!/bin/bash

# Python Library installation
if [[ -d "./dist" ]]; then
    pip3 uninstall dist/*.whl
    rm -rf dist build prode.egg-info
fi

if [[ ! -f "./requirements.txt" ]]; then
    pipreqs --force ./
fi

python3 setup.py bdist_wheel
pip3 install dist/*.whl
