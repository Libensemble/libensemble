#!/bin/bash

# Run this script between each repeated local run of the Balsam
#   job_control_hworld_balsam test in the base libensemble directory.
#   Besides ensuring that postgres and the generated Balsam db have the proper
#   permissions, this script flushes previous apps and jobs in the db before
#   submitting the test_balsam app/job script.
#
#   Most of this comes from scaling_tests/forces/balsam_local.sh

export EXE=$PWD/libensemble/tests/regression_tests/script_test_balsam.py
export NUM_WORKERS=2
export WORKFLOW_NAME=libe_test-balsam
export LIBE_WALLCLOCK=3
export THIS_DIR=$PWD
export SCRIPT_BASENAME=script_test_balsam

# Set proper permissions, initialize Balsam DB, activate DB
export BALSAM_DB_PATH='~/test-balsam'
sudo chown -R postgres:postgres /var/run/postgresql
sudo chmod a+w /var/run/postgresql
balsam init ~/test-balsam
sudo chmod -R 700 ~/test-balsam/balsamdb
source balsamactivate test-balsam

# Refresh DB
balsam rm apps --all --force
balsam rm jobs --all --force

# Submit script_test_balsam as app
balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

# Submit job based on script_test_balsam app
balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --wall-time-minutes $LIBE_WALLCLOCK --num-nodes 1 --ranks-per-node $((NUM_WORKERS+1)) --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" --url-in="local:/$THIS_DIR/*" --yes
