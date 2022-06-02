#!/bin/bash -x
#PBS -l walltime=00:30:00
#PBS -l select=4:ncpus=128
#PBS -A <project_code>

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the compute nodes in the allocation.

# Name of calling script

source ~/.bashrc

export EXE=<libE_calling_script.py>

# Communication Method
export COMMS="--comms local"

# Number of workers.
export NWORKERS="--nworkers 4"

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

# Activate conda environment
export PYTHONNOUSERSITE=1
conda activate $CONDA_ENV_NAME

cd $PBS_O_WORKDIR

# openMPI on edtb
export LD_LIBRARY_PATH=/lus/theta-fs0/software/edtb/openmpi/4.1.1/lib:$LD_LIBRARY_PATH
export PATH=/lus/theta-fs0/software/edtb/openmpi/4.1.1/bin:$PATH

# Launch libE
python $EXE $COMMS $NWORKERS > out.txt 2>&1
