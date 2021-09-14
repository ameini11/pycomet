#!/bin/bash

# Cleaning the run folder

find . -type d -name __pycache__ -exec rm -r {} \;
if [[ -d "./training" ]]; then
    rm -rf training postprocessing *.h5
fi

if [[ -d "./saving" ]]; then
    read -p "Do you want to delete the 'saving' folder? [y/n] " asw
    if [[ $asw = y* ]]; then
		rm -rf saving
	fi
fi

if [[ -d "./data" ]]; then
    read -p "Do you want to delete the 'data' folder? [y/n] " asw
    if [[ $asw = y* ]]; then
		rm -rf data
	fi
fi
