#!/bin/bash
#SBATCH -J libE_small_test
#SBATCH -A <myproject_g>
#SBATCH -C gpu
#SBATCH --time 10
#SBATCH --nodes 2

export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_EXACT=1
export SLURM_MEM_PER_NODE=0

python run_libe_forces.py --comms local --nworkers 8
