#!/bin/bash -x
#COBALT -t 00:30:00
#COBALT -n 129
#COBALT -q default
#COBALT -A <projectID>

# Script to launch libEnsemble using Balsam within Conda. Conda environment must be set up.

# Requires Balsam is installed and a database initialized (this can be the default database).

# To be run with central job management
# - Manager and workers run on one node (or a dedicated set of nodes).
# - Workers submit jobs to the rest of the nodes in the pool.

# Name of calling script
export EXE=run_libe_forces.py

# Number of workers.
export NUM_WORKERS=127

# Wallclock for libE job in minutes (supplied to Balsam - make at least several mins smaller than wallclock for this submission to ensure job is launched)
export LIBE_WALLCLOCK=25

# Name of working directory where Balsam places running jobs/output (inside the database directory)
export WORKFLOW_NAME=libe_workflow

# export SCRIPT_ARGS='' #Default No args
# export SCRIPT_ARGS=$(($LIBE_WALLCLOCK-5))
export SCRIPT_ARGS="--comms mpi --nworkers $NUM_WORKERS"

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

# Name of database
export DBASE_NAME=<dbase_name>

# Conda location - theta specific
# export PATH=/opt/intel/python/2017.0.035/intelpython35/bin:$PATH
# export LD_LIBRARY_PATH=~/.conda/envs/$CONDA_ENV_NAME/lib:$LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1 #Ensure environment isolated

export PMI_NO_FORK=1 # Required for python kills on Theta

export LIBE_PLOTS=true   # Require plot scripts (see at end)
export BALSAM_PLOTS=true # Require plot scripts (see at end)
export PLOT_DIR=..

# Activate conda environment
. activate $CONDA_ENV_NAME

# Unload Theta modules that may interfere with job monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

. balsamactivate $DBASE_NAME

# Make sure no existing apps/jobs
balsam rm apps --all --force
balsam rm jobs --all --force
wait
sleep 3

# Add calling script to Balsam database as app and job.
THIS_DIR=$PWD
SCRIPT_BASENAME=${EXE%.*}

# Running libE on one node - one manager and upto 63 workers
# NUM_NODES=1
# RANKS_PER_NODE=$((NUM_WORKERS+1)) # One node auto

# Multiple nodes
NUM_NODES=2
RANKS_PER_NODE=64

# All jobs
OUT_FILES_TO_RETURN="*.out *.txt *.log"

balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS --wall-time-minutes $LIBE_WALLCLOCK --num-nodes $NUM_NODES --ranks-per-node $RANKS_PER_NODE --url-out="local:/$THIS_DIR" --stage-out-files="${OUT_FILES_TO_RETURN}" --url-in="local:/$THIS_DIR/*" --yes

# Hyper-thread libE (note this will not affect HT status of user calcs - only libE itself)
# E.g. Running 255 workers and one manager on one libE node.
# balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS --wall-time-minutes $LIBE_WALLCLOCK --num-nodes $NUM_NODES --ranks-per-node $RANKS_PER_NODE --threads-per-core 4 --url-out="local:/$THIS_DIR" --stage-out-files="${OUT_FILES_TO_RETURN}" --url-in="local:/$THIS_DIR/*" --yes

#Run job
balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1

if [[ $LIBE_PLOTS = "true" ]]; then
  python $PLOT_DIR/plot_libe_calcs_util_v_time.py
  python $PLOT_DIR/plot_libe_runs_util_v_time.py
  python $PLOT_DIR/plot_libe_histogram.py

if [[ $BALSAM_PLOTS = "true" ]]; then
#   export MPLBACKEND=TkAgg
  python $PLOT_DIR/plot_util_v_time.py
  python $PLOT_DIR/plot_jobs_v_time.py
  python $PLOT_DIR/plot_waiting_v_time.py
fi

. balsamdeactivate
