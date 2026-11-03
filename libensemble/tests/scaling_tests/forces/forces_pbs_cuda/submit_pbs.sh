#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A <myproject>
#PBS -N libE_forces_pbs_cuda

export MPICH_GPU_SUPPORT_ENABLED=1
cd $PBS_O_WORKDIR

mpirun -np 5 python run_libe_forces.py
# python run_libe_forces.py --comms local --nworkers 5
