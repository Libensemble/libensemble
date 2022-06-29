#!/bin/bash -x
#BSUB -P <project code>
#BSUB -J libe_mproc
#BSUB -W 20
#BSUB -nnodes 4
#BSUB -alloc_flags "smt1"

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the nodes in the job available.

# Name of calling script-
export EXE=run_libe_forces.py

# Communication Method
export COMMS="--comms local"

# Number of workers.
export NWORKERS="--nworkers 4"

# Wallclock for libE. Slightly smaller than job wallclock
#export LIBE_WALLCLOCK=15 # Optional if pass to script

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

export LIBE_PLOTS=true # Require plot scripts in $PLOT_DIR (see at end)
export PLOT_DIR=..

# Need these if not already loaded
# module load python
# module load gcc/4.8.5

# Activate conda environment
export PYTHONNOUSERSITE=1
. activate $CONDA_ENV_NAME

# hash -d python # Check pick up python in conda env
hash -r # Check no commands hashed (pip/python...)

# Launch libE.
#python $EXE $NUM_WORKERS $LIBE_WALLCLOCK > out.txt 2>&1
python $EXE $COMMS $NWORKERS > out.txt 2>&1

if [[ $LIBE_PLOTS = "true" ]]; then
  python $PLOT_DIR/plot_libe_calcs_util_v_time.py
  python $PLOT_DIR/plot_libe_tasks_util_v_time.py
  python $PLOT_DIR/plot_libe_histogram.py
fi
