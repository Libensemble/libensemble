#!/bin/bash
#SBATCH -J libE_small_test
#SBATCH -A <myproject>
#SBATCH -C gpu
#SBATCH --time 10
#SBATCH --nodes 1

export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_EXACT=1

python run_libe_forces.py --comms local --nworkers 5
