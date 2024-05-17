#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A <myproject>

export MPICH_GPU_SUPPORT_ENABLED=1
cd $PBS_O_WORKDIR
python libE_calling_script.py -n 4
