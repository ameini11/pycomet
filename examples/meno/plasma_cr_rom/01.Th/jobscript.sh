#!/bin/bash -i

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --job-name=Th_r08c
#SBATCH --account=chess
#SBATCH --partition=chess-gpu
##SBATCH --dependency=afterok:1975667
#SBATCH --output=log.output_%j
#SBATCH --error=log.error_%j
##SBATCH --mail-user=zanardi3@illinois.edu
##SBATCH --mail-type=BEGIN,END

# Run python code/script
./../../pycomet train -d ./run08.c/
