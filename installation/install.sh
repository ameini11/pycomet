#!/bin/bash

# Python Library installation
bash installation/uninstall.sh

if [[ ! -f "./requirements.txt" ]]; then
    pipreqs --force ./
fi

python3 setup.py bdist_wheel
pip3 install dist/*.whl
