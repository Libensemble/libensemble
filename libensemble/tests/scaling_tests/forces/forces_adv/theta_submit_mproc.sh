#!/bin/bash -x
#COBALT -t 00:30:00
#COBALT -n 128
#COBALT -q default
#COBALT -A <project code>

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the nodes in the job available.

# Name of calling script
export EXE=run_libe_forces.py

# Communication Method
export COMMS="--comms local"

# Number of workers.
export NWORKERS="--nworkers 128"

# Wallclock for libE (allow clean shutdown)
#export LIBE_WALLCLOCK=25 # Optional if pass to script

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

# Conda location - theta specific
# export PATH=/opt/intel/python/2017.0.035/intelpython35/bin:$PATH
# export LD_LIBRARY_PATH=~/.conda/envs/<conda_env_name>/lib:$LD_LIBRARY_PATH
export PMI_NO_FORK=1 # Required for python kills on Theta

export LIBE_PLOTS=true # Require plot scripts in $PLOT_DIR (see at end)
export PLOT_DIR=..

# Unload Theta modules that may interfere with job monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

# Activate conda environment
export PYTHONNOUSERSITE=1
. activate $CONDA_ENV_NAME

# Launch libE
#python $EXE $NUM_WORKERS $LIBE_WALLCLOCK > out.txt 2>&1
#python $EXE $NUM_WORKERS > out.txt 2>&1
python $EXE $COMMS $NWORKERS > out.txt 2>&1

if [[ $LIBE_PLOTS = "true" ]]; then
  python $PLOT_DIR/plot_libe_calcs_util_v_time.py
  python $PLOT_DIR/plot_libe_runs_util_v_time.py
  python $PLOT_DIR/plot_libe_histogram.py
fi
