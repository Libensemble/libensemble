#!/bin/bash
#SBATCH -J libE_test_central
#SBATCH -N 5
#SBATCH -p knlall
#SBATCH -A <my_project>
#SBATCH -o tlib.%j.%N.out
#SBATCH -e tlib.%j.%N.error
#SBATCH -t 01:00:00

# Launch script for running in central mode with mpi4py.
#   libEnsemble will run on a dedicated node (or nodes).
#   The remaining nodes in the allocation will be dedicated to worker launched apps.
#   Use executor with auto-resources=True and central_mode=True.

# User to edit these variables
export EXE=libE_calling_script.py
export NUM_WORKERS=4
export I_MPI_FABRICS=shm:tmi

# Overcommit will allow ntasks up to the no. of contexts on one node (eg. 320 on Bebop)
srun --overcommit --ntasks=$(($NUM_WORKERS+1)) --nodes=1 python $EXE

# To use local mode instead of mpi4py (with parse_args())
# python calling_script.py --comms local --nworkers $NUM_WORKERS
