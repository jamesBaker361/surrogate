#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --account=m1248
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --constraint=haswell

conda activate testenv
srun python untitled.py
conda deactivate