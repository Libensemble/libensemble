#!/bin/bash
#SBATCH -J libE_simple
#SBATCH -A <myproject>
#SBATCH -p <partition_name>
#SBATCH -C <constraint_name>
#SBATCH --time 15
#SBATCH --nodes 2

# Usually either -p or -C above is used.

export MPICH_GPU_SUPPORT_ENABLED=1

python run_libe.py
