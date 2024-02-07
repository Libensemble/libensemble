#!/bin/bash -l
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -q <myqueue>
#PBS -A <myproject>

module use /soft/modulefiles
module load frameworks

export MPICH_GPU_SUPPORT_ENABLED=1
cd $PBS_O_WORKDIR

# 2 nodes - 12 sim workers (6 GPUs per node)
python libE_calling_script.py --comms local --nworkers 13
