#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --image=jlbaker361/routing-python
#SBATCH --volume=/global/u1/j/jamesbak/m1248:/m1248
#SBATCH --nodes=1
#SBATCH --time=00:12:00
#SBATCH --account=m1248
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_err/%j.err

srun shifter $@