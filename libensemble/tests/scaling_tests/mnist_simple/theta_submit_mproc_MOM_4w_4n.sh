#!/bin/bash -x
#COBALT -t 00:30:00
#COBALT -n 4
#COBALT -q debug-cache-quad
#COBALT -A CSC250STMS07

# Script to launch libEnsemble using multiprocessing within Conda. Conda environment must be set up.

# To be run with central job management
# - Manager and workers run on one node (or more if nec).
# - Workers submit jobs to the rest of the nodes in the pool.

# Constaint: - As set up - only uses one node (up to 63 workers) for libE.

# Name of calling script
export EXE=run_keras_mnist_simple.py

# Number of workers.
export NUM_WORKERS=4 # Can be passed as argument to calling script
export HDF5_USE_FILE_LOCKING=FALSE
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Name of Conda environment
export CONDA_ENV_NAME=again
export PYTHONNOUSERSITE=1 #Ensure environment isolated
export PMI_NO_FORK=1 # Required for python kills on Theta

module load miniconda-3/latest
module load gcc/9.3.0

# Must use mpich-intel-abi typically prepended by datascience module
export LD_LIBRARY_PATH=/opt/cray/pe/mpt/7.7.14/gni/mpich-intel-abi/16.0/lib:$LD_LIBRARY_PATH

# Activate conda environment
# source activate $CONDA_ENV_NAME
# source $CONDA_ENV_NAME/bin/activate
# Run from test directory, but env is in base job directory
conda activate $CONDA_ENV_NAME

# Unload Theta modules that may interfere with Balsam
module unload trackdeps
module unload darshan
module unload xalt

#python $EXE $NUM_WORKERS $LIBE_WALLCLOCK > out.txt 2>&1
python $EXE --comms local --nworkers $NUM_WORKERS > job_run_libe_forces.out 2>&1
