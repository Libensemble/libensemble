#!/usr/bin/env bash

#set -e #Exit on error - act. may need to source if interactive... use return or checks

#Run launcher either interactive or via script
#eg. Interactive - create session
#qsub -A CSC250STMS07 -n 1 -q debug-flat-quad -t 30 -I

# User inputs =================================================================================================

export JOB_NAME=run_mpi_libe_test

export LIBE_DIR=/home/shudson/balsam-libe-test/libensemble

#Location of main libensemble calling script
export WORK_DIR=$LIBE_DIR/code/tests/regression_tests

#export LIBE_APP=test_6-hump_camel_with_different_nodes_uniform_sample.py
export LIBE_APP=mpitest_six_hump_camel.py

export SIM_DIR=$LIBE_DIR/code/examples/sim_funcs
export SIM_APP=helloworld.py

export BALSAM_DIR=/home/shudson/balsam-libe-test/hpc-edge-service

export CLEAN_ALL=true #Cleans apps and jobs - overrides other clean commands
export CLEAN_JOBS=true

#For libensemble
export NUM_NODES=1
export RANKS_PER_NODE=4

#soft add +anaconda
#. activate balsam

# Set up conda env ============================================================================================

################ These are the ones I've now moved into serparate setup script - they are once per qsub.

#theta
################export PATH=/opt/intel/python/2017.0.035/intelpython35/bin:$PATH

#alias conda='/opt/intel/python/2017.0.035/intelpython35/bin/conda'

#Only needs to be done once on a system and then channel stored in ~/.condarc
#conda config --add channels intel

#Do once - then env stored
#conda create --name balsam intelpython3_full python=3
#cp  /opt/cray/pe/mpt/7.6.0/gni/mpich-intel-abi/16.0/lib/libmpi* ~/.conda/envs/balsam/lib/

################export LD_LIBRARY_PATH=~/.conda/envs/balsam/lib:$LD_LIBRARY_PATH

#My addition - takes non-conda local dirs out of sys path - to help isolate conda
################export PYTHONNOUSERSITE=1

################. activate balsam #Should be a check here.

#Note remember first time will need to pip install balsam (hpc-edge-service/)
#can do on logins if load balsam conda env.

# Set up job ==================================================================================================

#NOTE: This does not have to be done in the interactive session - as long as balsam conda env is loaded
#Prob make this sep. script - callable from the run script if nec.

ABORT=false
SET_JOB=false

SIM_APP_NAME=${SIM_APP%.*}
LIBE_APP_NAME=${LIBE_APP%.*}

if [[ $CLEAN_ALL = "true" ]]
then
  balsam rm apps --all
  balsam rm jobs --all
  
  #Add apps - includes libensemble and any apps it will be launching
  
  #set -x  
  balsam app --name $SIM_APP_NAME --exec $SIM_DIR/$SIM_APP --desc "Run $SIM_APP_NAME" || ABORT=true
  balsam app --name $LIBE_APP_NAME --exec $WORK_DIR/$LIBE_APP --desc "Run $LIBE_APP_NAME" || ABORT=true  
  #set +x
  
  SET_JOB=true
  
  #Add jobs - libensemble
  #balsam job --name run_mpi_libe_test --workflow libe_workflow --application mpitest_libe --wall-min 1 --num-nodes 1 --ranks-per-node 2 || ABORT=true
    
elif [[ $CLEAN_JOBS = "true" ]]
then
  balsam rm jobs --all
  SET_JOB=true
  
  #Note - application name must match with existing app - is there a balsam test for this???
  #Add jobs - libensemble
  #balsam job --name run_mpi_libe_test --workflow libe_workflow --application mpitest_libe --wall-min 1 --num-nodes 1 --ranks-per-node 2 || ABORT=true
    
fi

if [[ $ABORT = "false" ]]
then
  if [[ $SET_JOB = "true" ]]
  then
    
    #OUT_DIR="local:$WORK_DIR"
    set -x
    #Add jobs - libensemble - hardcoded path....
    balsam job --name $JOB_NAME --workflow libe_workflow --application $LIBE_APP_NAME \
               --wall-min 1 --num-nodes $NUM_NODES --ranks-per-node $RANKS_PER_NODE \
               --url-out="local:$WORK_DIR" \
               --stage-out-files="$JOB_NAME*" || ABORT=true   
    set +x 
  fi  
else
  echo -e "\n*** Aborting - before job set - check apps ***"
fi

LAUNCH_JOB=false 
if [[ $ABORT = "false" ]]
then
  echo -e "\nListing apps:"
  balsam ls apps

  echo -e "\nListing jobs:"
  balsam ls jobs
  
  read -r -p "Continue with launcher? [y/N] " response
  case "$response" in
      [yY][eE][sS]|[yY]) 
           LAUNCH_JOB=true
          ;;
      *)
          echo -e "\n*** Aborting - Abort issued by user ***"
          ;;
  esac

else
  echo -e "\n*** Aborting - check job/s ***"
fi


if [[ $LAUNCH_JOB = "true" ]]
then
  LAUNCH_COMMAND="balsam launcher --consume-all"
  echo -e "\n$LAUNCH_COMMAND"
  $LAUNCH_COMMAND   
fi




