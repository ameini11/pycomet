#!/bin/bash -i

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=datagen
#SBATCH --account=chess
#SBATCH --partition=chess
##SBATCH --dependency=afterok:2002687
#SBATCH --output=log.output_%j
#SBATCH --error=log.error_%j
##SBATCH --mail-user=zanardi3@illinois.edu
##SBATCH --mail-type=BEGIN,END

# Run python code/script
./../../pycomet generate -d ./
mv data data.v8