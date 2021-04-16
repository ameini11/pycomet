#!/bin/bash

# Python Library installation
pip3 uninstall dist/*.whl
rm -rf dist build prode.egg-info
python3 setup.py bdist_wheel
pip3 install dist/*.whl

