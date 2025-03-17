#!/bin/bash -l
#PBS -l select=2:mpiprocs=16
#PBS -l walltime=00:15:00
#PBS -q bdwall
#PBS -A [project]
#PBS -N libE_example


cd $PBS_O_WORKDIR
module load openmpi

mpirun -n 16 --ppn 8 python run_libe_example.py
