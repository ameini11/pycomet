#!/bin/bash

# Cleaning the run folder
if [[ -d "./__pycache__" ]] || [[ -d "./training" ]]; then
    rm -rf training postprocessing __pycache__ *.h5
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