#!/bin/bash -i

# Activate the Conda environment where PyCOMET is installed
conda activate pycomet

# Specify a custom PyCOMET configuration file (optional)
export PYCOMET_CFGFILE="./pcmcfg.json"

# Launch each Python script and redirect logs
./ode/main.py > log.ode
./sur/main.py > log.sur
