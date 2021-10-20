#!/bin/bash -x
#SBATCH --job-name=libE-test
#SBATCH --account=STARTUP-USERNAME
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00

# Make sure conda and your environment are loaded and activated before sbatch

module load gcc
module load cuda/11.0.2-4szlv2t

unset I_MPI_FABRICS
export I_MPI_FABRICS_LIST=tmi,tcp
export I_MPI_FALLBACK=1

export EXE=run_libe_ddmd.py
export NUM_WORKERS=2
export MANAGER_NODE=false # true = Manager has a dedicated node (assign one extra)
export USE_NODE_LIST=true # If false, allow libE to determine node_list from environment.

# Activate conda environment
export PYTHONNOUSERSITE=1

#-----------------------------------------------------------------------------
# Work out distribution
if [[ $MANAGER_NODE = "true" ]]; then
  WORKER_NODES=$(($SLURM_NNODES-1))
else
  WORKER_NODES=$SLURM_NNODES
fi

if [[ $NUM_WORKERS -ge $WORKER_NODES ]]; then
  SUB_NODE_WORKERS=true
  WORKERS_PER_NODE=$(($NUM_WORKERS/$WORKER_NODES))
else
  SUB_NODE_WORKERS=false
  NODES_PER_WORKER=$(($WORKER_NODES/$NUM_WORKERS))
fi;
#-----------------------------------------------------------------------------

# A little useful information
echo -e "Manager process running on: $HOSTNAME"
echo -e "Directory is:  $PWD"

# Generate a node list with 1 node per line:
srun hostname | sort -u > node_list

# Add manager node to machinefile
head -n 1 node_list > machinefile.$SLURM_JOBID

# Add worker nodes to machinefile
if [[ $SUB_NODE_WORKERS = "true" ]]; then
  awk -v repeat=$WORKERS_PER_NODE '{for(i=0;i<repeat;i++)print}' node_list \
  >>machinefile.$SLURM_JOBID
else
  awk -v patt="$NODES_PER_WORKER" 'NR % patt == 1' node_list \
  >> machinefile.$SLURM_JOBID
fi;

if [[ $USE_NODE_LIST = "false" ]]; then
  rm node_list
  wait
fi;

# Put in a timestamp
echo Starting execution at: `date`

# To use srun
export SLURM_HOSTFILE=machinefile.$SLURM_JOBID

# The "arbitrary" flag should ensure SLURM_HOSTFILE is picked up
# cmd="srun --ntasks $(($NUM_WORKERS+1)) -m arbitrary python $EXE"
cmd="srun --ntasks $(($NUM_WORKERS+1)) -m arbitrary python $EXE $LIBE_WALLCLOCK"

echo The command is: $cmd
echo End PBS script information.
echo All further output is from the process being run and not the script.\n\n $cmd

$cmd

# Print the date again -- when finished
echo Finished at: `date`
