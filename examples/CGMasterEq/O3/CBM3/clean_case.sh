#!/bin/bash

# Python Library installation
if [[ -d "./__pycache__" ]] || [[ -d "./training" ]]; then
    rm -rf training postprocessing __pycache__ *.h5 data
    read -p "Do you want to delete the 'saving' folder? [y/n] " asw
    if [[ $asw = y* ]]; then
		rm -rf saving
	fi
fi
