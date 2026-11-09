#!/bin/bash
#SBATCH -J libE_forces_slurm_cuda
#SBATCH -A <myproject>
#SBATCH -C gpu
#SBATCH --time 00:15:00
#SBATCH --nodes 1

export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_EXACT=1

srun -n 5 python run_libe_forces.py
# python run_libe_forces.py --comms local --nworkers 5
