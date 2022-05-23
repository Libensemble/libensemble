#!/bin/bash

export EXE=script_test_balsam_hworld.py
export NUM_WORKERS=2
export WORKFLOW_NAME=libe_test-balsam
export LIBE_WALLCLOCK=3
export SCRIPT_BASENAME=script_test_balsam_hworld
export BALSAM_DB_PATH=$HOME/test-balsam
source balsamactivate test-balsam

# Refresh DB
balsam rm apps --all --force
balsam rm jobs --all --force

# Submit script_test_balsam_hworld as app
balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

# Submit job based on script_test_balsam_hworld app
balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --wall-time-minutes $LIBE_WALLCLOCK --num-nodes 1 --ranks-per-node $((NUM_WORKERS+1)) --url-out="local:/$PWD" --stage-out-files="*.out *.txt *.log" --url-in="local:/$PWD/*" --yes
