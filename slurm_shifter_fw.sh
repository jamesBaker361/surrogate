#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --image=jlbaker361/frank-wolfe
#SBATCH --volume=/global/u1/j/jamesbak/m1248:/m1248
#SBATCH --nodes=1
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --mem=32000
#SBATCH --account=m1248
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_err/%j.err

srun shifter $@