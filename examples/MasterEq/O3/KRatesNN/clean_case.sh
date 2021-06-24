#!/bin/bash

# Cleaning the run folder
if [[ -d "./__pycache__" ]]; then

    rm -rf data plots __pycache__

fi
