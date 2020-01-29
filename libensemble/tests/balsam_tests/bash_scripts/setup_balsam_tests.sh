#!/usr/bin/env bash

set -e #Exit on error

# User inputs =================================================================================================

export LIBE_DIR=/home/shudson/balsam-libe-test/libensemble
#export BALSAM_DIR=/home/shudson/balsam-libe-test/hpc-edge-service

#Location of main calling scripts
export WORK_DIR=$LIBE_DIR/code/tests/balsam_tests

export SIM_DIR=$LIBE_DIR/code/examples/sim_funcs
export SIM_APP=helloworld.py

#For host code
export NUM_NODES=1
export RANKS_PER_NODE=4

export JOB_LIST="test_balsam_1__runjobs.py \
                 test_balsam_2__workerkill.py \
                 test_balsam_3__managerkill.py"

# Set up job ==================================================================================================

#NOTE: This does not have to be done in the interactive session - as long as balsam conda env is loaded

#Check clean
balsam rm apps --all
balsam rm jobs --all

#Add user apps - eg helloworld.py
SIM_APP_NAME=${SIM_APP%.*}
balsam app --name $SIM_APP_NAME --exec $SIM_DIR/$SIM_APP --desc "Run $SIM_APP_NAME"

#Add apps and jobs for tests
for LIBE_APP in $JOB_LIST
do
  LIBE_APP_NAME=${LIBE_APP%.*}
  balsam app --name $LIBE_APP_NAME --exec $WORK_DIR/$LIBE_APP --desc "Run $LIBE_APP_NAME"

  #Add jobs
  balsam job --name job_$LIBE_APP_NAME --workflow libe_workflow --application $LIBE_APP_NAME \
             --wall-min 1 --num-nodes $NUM_NODES --ranks-per-node $RANKS_PER_NODE \
             --url-out="local:$WORK_DIR" \
             --stage-out-files="${JOB_NAME}*" \
             --yes

  #Add dependency so jobs run one at a time ....
  # *** Currently all jobs added will run simultaneously - kills may be an issue there

done

echo -e "\nListing apps:"
balsam ls apps

echo -e "\nListing jobs:"
balsam ls jobs

#Run launcher in either interactive session or via script
echo -e "\nTo launch jobs run: balsam launcher --consume-all"
