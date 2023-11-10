#!/bin/bash
#SBATCH -J libE_small_test
#SBATCH -A csc314
#SBATCH -p batch
#SBATCH --time 20
#SBATCH --nodes 1

export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_EXACT=1
export SLURM_MEM_PER_NODE=0

export PLATFORM=frontier

. ./run_timing_study.sh $PLATFORM
