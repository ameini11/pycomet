#!/bin/bash -i

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=datagen
#SBATCH --account=chess
#SBATCH --partition=chess
#SBATCH --output=log.output_%j
#SBATCH --error=log.error_%j
##SBATCH --dependency=afterok:2002687
##SBATCH --mail-user=zanardi3@illinois.edu
##SBATCH --mail-type=BEGIN,END

# Activate the Conda environment where PyCOMET is installed
conda activate pycomet

# Specify a custom PyCOMET configuration file (optional)
export PYCOMET_CFGFILE="./../../pcmcfg.json"

# Run PyCOMET script
pycomet save -d ./
