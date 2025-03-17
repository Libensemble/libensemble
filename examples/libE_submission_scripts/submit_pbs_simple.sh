#!/bin/bash -l
#PBS -l select=2
#PBS -l walltime=00:15:00
#PBS -q <queue_name>
#PBS -A <myproject>

# We selected 2 nodes - now running with 8 workers.
export MPICH_GPU_SUPPORT_ENABLED=1
cd $PBS_O_WORKDIR
python libE_calling_script.py -n 8
