#!/bin/bash -l
#PBS -l select=4
#PBS -l walltime=00:15:00
#PBS -q bdwall
#PBS -A [project]
#PBS -N libE_example

cd $PBS_O_WORKDIR
# Choose MPI backend. Note that the built mpi4py in your environment should match.
module load oneapi/mpi
# module load openmpi

python run_libe_example.py -n 16
