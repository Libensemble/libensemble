#!/bin/bash
#SBATCH -J libE_test
#SBATCH -N 4
#SBATCH -p [queue]
#SBATCH -A <my_project>
#SBATCH -o tlib.%j.%N.out
#SBATCH -e tlib.%j.%N.error
#SBATCH -t 01:00:00

# Launch script that runs in distributed mode with mpi4py.
#   Workers are evenly spread over nodes and manager added to the first node.
#   Requires even distribution - either multiple workers per node or nodes per worker
#   Option for manager to have a dedicated node.
#   Use of MPI Executor will ensure workers co-locate tasks with workers
#   If node_list file is kept, this informs libe of resources. Else, libe auto-detects.

# User to edit these variables
export EXE=libE_calling_script.py
export NUM_WORKERS=4
export MANAGER_NODE=false # true = Manager has a dedicated node (assign one extra)
export USE_NODE_LIST=true # If false, allow libE to determine node_list from environment.

# Sometimes may be necessary
# As libE shares nodes with user applications allow fallback if contexts overrun.
# unset I_MPI_FABRICS
# export I_MPI_FABRICS_LIST=tmi,tcp
# export I_MPI_FALLBACK=1

# If using in calling script (After N mins manager kills workers and exits cleanly)
export LIBE_WALLCLOCK=55

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
  awk -v repeat=$WORKERS_PER_NODE '{for(i=0; i<repeat; i++)print}' node_list \
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
