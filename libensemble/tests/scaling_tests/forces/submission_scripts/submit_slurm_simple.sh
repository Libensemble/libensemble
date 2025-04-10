#!/bin/bash
#SBATCH -J libE_simple
#SBATCH -A <myproject>
#SBATCH -p <partition_name>
#SBATCH -C <constraint_name>
#SBATCH --time 10
#SBATCH --nodes 2

# Usually either -p or -C above is used.

# On some SLURM configurations, these ensure runs can share nodes
export SLURM_EXACT=1
export SLURM_MEM_PER_NODE=0

python run_libe_forces.py -n 9
