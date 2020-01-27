#!/bin/bash

# Run this script between each repeated local run of the Balsam
#   job_control_hworld_balsam test in the base libensemble directory.
#   Besides ensuring that postgres and the generated Balsam db have the proper
#   permissions, this script flushes previous apps and jobs in the db before
#   submitting the test_balsam_hworld app/job script.
#
#   Most of this comes from scaling_tests/forces/balsam_local.sh

# Can't run this line in calling Python file. Balsam installation hasn't been
#   noticed by the Python runtime yet.
python -c 'from libensemble.tests.regression_tests.common import modify_Balsam_pyCoverage; modify_Balsam_pyCoverage()'
export EXE=$PWD/libensemble/tests/regression_tests/script_test_balsam_hworld.py
export NUM_WORKERS=2
export WORKFLOW_NAME=libe_test-balsam
export LIBE_WALLCLOCK=3
export THIS_DIR=$PWD
export SCRIPT_BASENAME=script_test_balsam_hworld

# Set proper permissions, initialize Balsam DB, activate DB
export BALSAM_DB_PATH=$HOME/test-balsam
sudo chown -R postgres:postgres /var/run/postgresql
sudo chmod a+w /var/run/postgresql
balsam init $HOME/test-balsam
sudo chmod -R 700 $HOME/test-balsam/balsamdb
source balsamactivate test-balsam

# Refresh DB
balsam rm apps --all --force
balsam rm jobs --all --force

# Submit script_test_balsam_hworld as app
balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

# Submit job based on script_test_balsam_hworld app
balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --wall-time-minutes $LIBE_WALLCLOCK --num-nodes 1 --ranks-per-node $((NUM_WORKERS+1)) --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" --url-in="local:/$THIS_DIR/*" --yes
