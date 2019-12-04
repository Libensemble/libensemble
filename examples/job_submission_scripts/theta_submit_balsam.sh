#!/bin/bash -x
#COBALT -t 30
#COBALT -O libE_MPI_balsam
#COBALT -n 5 # No. nodes
#COBALT -q debug-flat-quad # Up to 8 nodes only
##COBALT -q default # For large jobs >=128 nodes
##COBALT -A <project code>

# Script to launch libEnsemble using Balsam.
#   Assumes Conda environment is set up.
#   Requires Balsam is installed and a database initialized.

# To be run with central job management
# - Manager and workers run on one node (or a dedicated set of nodes).
# - Workers submit jobs to the rest of the nodes in the pool.

# Constaint: - As set up - only uses one node (up to 63 workers) for libE.
#   To use more, modifiy "balsam job" line to use hyper-threading
#   and/or more than one node for libE.

# Name of calling script
export EXE=libE_calling_script.py

# Number of workers.
export NUM_WORKERS=4

# Balsam wall-clock in minutes - make few mins smaller than batch wallclock
export BALSAM_WALLCLOCK=25

# Name of working directory within database where Balsam places running jobs/output
export WORKFLOW_NAME=libe_workflow

# Wall-clock in mins for libE (allow clean shutdown).
# Script must be set up to receive as argument.
export LIBE_WALLCLOCK=$(($BALSAM_WALLCLOCK-3))

# libEnsemble calling script arguments (some alternatives shown)

# No args. All defined in calling script
# export SCRIPT_ARGS=''

# If calling script is using utils.parse_args()
# export SCRIPT_ARGS="--comms mpi --nworkers $NUM_WORKERS

# If calling script takes wall-clock as positional arg and uses utils.parse_args()
export SCRIPT_ARGS="$LIBE_WALLCLOCK --comms mpi --nworkers $NUM_WORKERS"

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

# Name of database
export DBASE_NAME=<dbase_name>  # default - to use default database.

# Conda location - theta specific
export PATH=/opt/intel/python/2017.0.035/intelpython35/bin:$PATH
export LD_LIBRARY_PATH=~/.conda/envs/$CONDA_ENV_NAME/lib:$LD_LIBRARY_PATH

export PYTHONNOUSERSITE=1 # Ensure environment isolated

export PMI_NO_FORK=1 # Required for python kills on Theta

# Activate conda environment
. activate $CONDA_ENV_NAME

# Unload Theta modules that may interfere with job monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

# Activate Balsam database
. balsamactivate $DBASE_NAME

# Currently need atleast one DB connection per worker (for postgres).
if [[ $NUM_WORKERS -gt 128 ]]
then
   #Add a margin
   echo -e "max_connections=$(($NUM_WORKERS+10)) #Appended by submission script" \
   >> $BALSAM_DB_PATH/balsamdb/postgresql.conf
fi
wait

# Make sure no existing apps/jobs registered to database
balsam rm apps --all --force
balsam rm jobs --all --force
wait
sleep 3

# Add calling script to Balsam database as app and job.
THIS_DIR=$PWD
SCRIPT_BASENAME=${EXE%.*}

balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

# Running libE on one node - one manager and upto 63 workers
balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME \
       --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS \
       --wall-time-minutes $BALSAM_WALLCLOCK \
       --num-nodes 1 --ranks-per-node $((NUM_WORKERS+1)) \
       --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" \
       --url-in="local:/$THIS_DIR/*" --yes

# Hyper-thread libE (note this will not affect HT status of user calcs - only libE itself)
# Running 255 workers and one manager on one libE node.
# balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS --wall-time-minutes $BALSAM_WALLCLOCK  --num-nodes 1 --ranks-per-node 256 --threads-per-core 4 --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" --url-in="local:/$THIS_DIR/*" --yes

# Multiple nodes for libE
# Running 127 workers and one manager - launch script on 129 nodes (if one node per worker)
# balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS --wall-time-minutes $BALSAM_WALLCLOCK  --num-nodes 2 --ranks-per-node 64 --url-out="local:/$THIS_DIR"  --stage-out-files="*.out *.txt *.log" --url-in="local:/$THIS_DIR/*" --yes

# Run job
balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1

# Deactivate Balsam database
. balsamdeactivate
