#!/bin/bash -x
#COBALT -t 00:30:00
#COBALT -O libE_mproc_MOM
#COBALT -n 4
#COBALT -q debug-flat-quad # Up to 8 nodes only
##COBALT -q default # For large jobs >=128 nodes
##COBALT -A <project code>

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the compute nodes in the allocation.

# Name of calling script
export EXE=libE_calling_script.py

# Communication Method
export COMMS="--comms local"

# Number of workers.
export NWORKERS="--nworkers 4"

# Wallclock for libE (allow clean shutdown)
export LIBE_WALLCLOCK=25 # Optional if pass to script

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

# Conda location - theta specific
export PATH=/opt/intel/python/2017.0.035/intelpython35/bin:$PATH
export LD_LIBRARY_PATH=~/.conda/envs/$CONDA_ENV_NAME/lib:$LD_LIBRARY_PATH
export PMI_NO_FORK=1 # Required for python kills on Theta

# Unload Theta modules that may interfere with job monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

# Activate conda environment
export PYTHONNOUSERSITE=1
. activate $CONDA_ENV_NAME

# Launch libE
# python $EXE $NUM_WORKERS > out.txt 2>&1  # No args. All defined in calling script
# python $EXE $COMMS $NWORKERS > out.txt 2>&1  # If calling script is using parse_args()
python $EXE $LIBE_WALLCLOCK $COMMS $NWORKERS > out.txt 2>&1 # If calling script takes wall-clock as positional arg.
