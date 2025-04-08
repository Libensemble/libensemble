#!/bin/bash
#SBATCH -J libE_small_test
#SBATCH -A <myproject>
#SBATCH -C gpu
#SBATCH --time 10
#SBATCH --nodes 1

# This script is using GPU partition
export MPICH_GPU_SUPPORT_ENABLED=1

# One worker for generator and 4 for sims (one GPU each)
python libe_calling_script.py -n 5

# Or if libE_specs option gen_on_manager=True
# python libe_calling_script.py -n 4
