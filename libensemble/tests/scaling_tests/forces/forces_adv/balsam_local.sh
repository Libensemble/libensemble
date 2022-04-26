# Script for running with Balsam on a local system.

# NOTE: As of libEnsemble v0.9.0 there is a more up to date Balsam available.
# This version is only recommended if that one cannot be accessed or application
# kills are required.

# You need to have followed the instructions to install balsam and set-up/activate a database.
# https://github.com/balsam-alcf/balsam

# The running jobs can be seen inside the setup database dir <DIR>/data/libe_workflow/

# Name of calling script
export EXE=run_libe_forces.py

# Number of workers.
export NUM_WORKERS=2

# Name of working directory where Balsam places running jobs/output (inside the database directory)
export WORKFLOW_NAME=libe_workflow

export SCRIPT_ARGS=$NUM_WORKERS

export LIBE_WALLCLOCK=5 # Balsam timeout in mins

# Add calling script to Balsam database as app and job.
export THIS_DIR=$PWD
export SCRIPT_BASENAME=${EXE%.*}

# Delete any apps/jobs in Balsam
balsam rm apps --all --force
balsam rm jobs --all --force

# Register your libEnsemble calling script as an app.
balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

# Register as a job
balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS --wall-time-minutes $LIBE_WALLCLOCK --num-nodes 1 --ranks-per-node $((NUM_WORKERS+1)) --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" --url-in="local:/$THIS_DIR/*" --yes

#Run job
balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1
