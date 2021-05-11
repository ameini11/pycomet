#!/bin/bash

# Python Library installation
if [[ -d "./__pycache__" ]] || [[ -d "./training" ]]
then
    rm -rf training postprocessing __pycache__ *.h5
fi


