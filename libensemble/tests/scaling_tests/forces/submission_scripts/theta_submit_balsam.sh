#!/bin/bash -x
#COBALT -t 00:30:00
#COBALT -O libE_forces_MPI_balsam
#COBALT -n 129
#COBALT -q default
#COBALT -A <projectID>

# Script to launch libEnsemble using Balsam.
#   Assumes Conda environment is set up.
#   Requires Balsam is installed and a database initialized.

# To be run with central job management
# - Manager and workers run on one node (or a dedicated set of nodes).
# - Workers submit tasks to the rest of the nodes in the pool.

# Name of calling script
export EXE=run_libe_forces.py

# Number of workers.
export NUM_WORKERS=127

# Number of nodes to run libE
export LIBE_NODES=2

# Wallclock for libE job in minutes (supplied to Balsam - make at least several mins smaller than wallclock for this submission to ensure job is launched)
export LIBE_WALLCLOCK=25

# Name of working directory where Balsam places running jobs/output (inside the database directory)
export WORKFLOW_NAME=libe_workflow

# If user script takes ``wallclock_max`` argument.
# export SCRIPT_ARGS=$(($LIBE_WALLCLOCK-3))
export SCRIPT_ARGS=""

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

# Name of database
export BALSAM_DB_NAME=<dbase_name>

export LIBE_PLOTS=true   # Require plot scripts (see at end)
export BALSAM_PLOTS=true # Require plot scripts (see at end)
export PLOT_DIR=..

# Required for killing tasks from workers on Theta
export PMI_NO_FORK=1

# Unload Theta modules that may interfere with task monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

# Obtain Conda PATH from miniconda-3/latest module
CONDA_DIR=/soft/datascience/conda/miniconda3/latest/bin

# Ensure environment isolated
export PYTHONNOUSERSITE=1

# Activate conda environment
source $CONDA_DIR/activate $CONDA_ENV_NAME

# Activate Balsam database
source balsamactivate $BALSAM_DB_NAME

# Make sure no existing apps/jobs
balsam rm apps --all --force
balsam rm jobs --all --force
wait
sleep 3

# Add calling script to Balsam database as app and job.
export THIS_DIR=$PWD
export SCRIPT_BASENAME=${EXE%.*}

# Multiple nodes
export LIBE_PROCS=$((NUM_WORKERS+1))  # Manager and workers
export PROCS_PER_NODE=$((LIBE_PROCS/LIBE_NODES))  # Must divide evenly

balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME \
           --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS \
           --wall-time-minutes $LIBE_WALLCLOCK \
           --num-nodes $LIBE_NODES --ranks-per-node $PROCS_PER_NODE \
           --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" \
           --url-in="local:/$THIS_DIR/*" --yes

# Hyper-thread libE (note this will not affect HT status of user calcs - only libE itself)
# E.g. Running 255 workers and one manager on one libE node.
# balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME \
#            --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS \
#            --wall-time-minutes $LIBE_WALLCLOCK \
#            --num-nodes 1 --ranks-per-node 256 --threads-per-core 4 \
#            --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" \
#            --url-in="local:/$THIS_DIR/*" --yes

#Run job
balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1

if [[ $LIBE_PLOTS = "true" ]]; then
  python $PLOT_DIR/plot_libe_calcs_util_v_time.py
  python $PLOT_DIR/plot_libe_tasks_util_v_time.py
  python $PLOT_DIR/plot_libe_histogram.py

if [[ $BALSAM_PLOTS = "true" ]]; then
  python $PLOT_DIR/plot_util_v_time.py
  python $PLOT_DIR/plot_jobs_v_time.py
  python $PLOT_DIR/plot_waiting_v_time.py
fi

wait
source balsamdeactivate
