#!/bin/bash -x
#COBALT -t 30
#COBALT -O libE_MPI_balsam
#COBALT -n 5
#COBALT -q debug-flat-quad # Up to 8 nodes only # Use default for >=128 nodes
#COBALT -A <project code>

# Script to launch libEnsemble using Balsam.
#   Assumes Conda environment is set up.
#   Requires Balsam is installed and a database initialized.

# To be run with central job management
# - Manager and workers run on one node (or a dedicated set of nodes).
# - Workers submit tasks to the rest of the nodes in the pool.

# Name of calling script
export EXE=libE_calling_script.py

# Number of workers.
export NUM_WORKERS=4

# Number of nodes to run libE
export LIBE_NODES=1

# Balsam wall-clock in minutes - make few mins smaller than batch wallclock
export BALSAM_WALLCLOCK=25

# Name of working directory within database where Balsam places running jobs/output
export WORKFLOW_NAME=libe_workflow

# Wall-clock in mins for libE (allow clean shutdown).
# Script must be set up to receive as argument.
export LIBE_WALLCLOCK=$(($BALSAM_WALLCLOCK-3))

# libEnsemble calling script arguments (some alternatives shown)

# No args. All defined in calling script
export SCRIPT_ARGS=''

# If calling script takes wall-clock as positional argument.
# export SCRIPT_ARGS="$LIBE_WALLCLOCK"

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

# Name of database
export BALSAM_DB_NAME=<dbase_name>  # default - to use default database.

# Required for killing tasks from workers on Theta
export PMI_NO_FORK=1

# Unload Theta modules that may interfere with job monitoring/kills
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

# Currently need at least one DB connection per worker (for postgres).
if [[ $NUM_WORKERS -gt 100 ]]
then
   #Add a margin
   echo -e "max_connections=$(($NUM_WORKERS+20)) #Appended by submission script" \
   >> $BALSAM_DB_PATH/balsamdb/postgresql.conf
fi
wait

# Make sure no existing apps/tasks registered to database
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

# Run job
balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1

wait
source balsamdeactivate
