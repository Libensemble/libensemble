#!/bin/bash
#SBATCH -J libE_test_central
#SBATCH -N 5
#SBATCH -p RM
#SBATCH -A <my_project>
#SBATCH -o tlib.%j.%N.out
#SBATCH -e tlib.%j.%N.error
#SBATCH -t 00:30:00

# Launch script for running in central mode with mpi4py.
#   libEnsemble will run on a dedicated node (or nodes).
#   The remaining nodes in the allocation will be dedicated to worker launched apps.
#   Initialize Executor with auto-resources=True and central_mode=True.

# User to edit these variables
export EXE=libE_calling_script.py
export NUM_WORKERS=4

mpirun -np $(($NUM_WORKERS+1)) -ppn $(($NUM_WORKERS+1)) python $EXE

# To use local mode instead of mpi4py (with parse_args())
# python $EXE -n $NUM_WORKERS
