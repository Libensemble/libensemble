#!/bin/bash -x
#BSUB -P csc314
#BSUB -J libe_mproc
#BSUB -W 20
#BSUB -nnodes 4
#BSUB -alloc_flags "smt1"

# Script to launch libEnsemble using multiprocessing

# Name of calling script-
export EXE=run_libe_forces.py

# Wallclock for libE. Slightly smaller than job wallclock
export LIBE_WALLCLOCK=15

# Name of Conda environment 
export CONDA_ENV_NAME=libe-gcc

export LIBE_PLOTS=true   # Require plot scripts (see at end)
export PLOT_DIR=..

# Need these if not already loaded
# module load python
# module load gcc/4.8.5

# Activate conda environment
export PYTHONNOUSERSITE=1
. activate $CONDA_ENV_NAME

# hash -d python # Check pick up python in conda env
hash -r # Check no commands hashed (pip/python...)

python $EXE $LIBE_WALLCLOCK > out.txt 2>&1

if [[ $LIBE_PLOTS = "true" ]]; then
  python $PLOT_DIR/plot_libe_calcs_util_v_time.py
  python $PLOT_DIR/plot_libE_histogram.py
fi
