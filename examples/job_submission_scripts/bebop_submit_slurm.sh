#!/bin/bash
#SBATCH -J libE_test
#SBATCH -N 4
#SBATCH -p knlall
##SBATCH -A <my_project>
#SBATCH -o tlib.%j.%N.out 
#SBATCH -e tlib.%j.%N.error 
#SBATCH -t 01:00:00 

#Launch script that evenly spreads workers and adds manager to the first node.
#Requires even distribution - either multiple workers per node or nodes per worker
#Now option for manager to have a dedicated node.

#User to edit these variables
export EXE=libE_calling_script.py
export NUM_WORKERS=4
export MANAGER_NODE=false #true = Manager has a dedicated node (use one extra node for SBATCH -N)

unset I_MPI_FABRICS
export I_MPI_FABRICS_LIST=tmi,tcp
export I_MPI_FALLBACK=1


#If using in calling script (After N mins manager kills workers and timing.dat created.)
export LIBE_WALLCLOCK=55

#---------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------
#Test
echo -e "Slurm job ID: $SLURM_JOBID"
 	
#cd $PBS_O_WORKDIR
cd $SLURM_SUBMIT_DIR

# A little useful information for the log file...
echo -e "Master process running on: $HOSTNAME"
echo -e "Directory is:  $PWD"

#Generate a node list with 1 node per line:
srun hostname | sort -u > node_list

#Generate list of nodes for workers
if [[ $MANAGER_NODE = "true" ]]; then
  tail -n +2 node_list > worker_list  
else
  cp node_list worker_list
fi

#Add manager node to machinefile
head -n 1 node_list > machinefile.$SLURM_JOBID

#Add worker nodes to machinefile
if [[ $SUB_NODE_WORKERS = "true" ]]; then
  awk -v repeat=$WORKERS_PER_NODE '{for(i=0;i<repeat;i++)print}' worker_list >>machinefile.$SLURM_JOBID
else
  awk -v patt="$NODES_PER_WORKER" 'NR % patt == 1' worker_list >> machinefile.$SLURM_JOBID
fi;

# Put in a timestamp
echo Starting executation at: `date`

#To use srun
export SLURM_HOSTFILE=machinefile.$SLURM_JOBID
# cmd="srun --ntasks $(($NUM_WORKERS+1)) -m arbitrary python $EXE"
cmd="srun --ntasks $(($NUM_WORKERS+1)) -m arbitrary python $EXE $LIBE_WALLCLOCK"

echo The command is: $cmd 
echo End PBS script information. 
echo All further output is from the process being run and not the pbs script.\n\n $cmd # Print the date again -- when finished 

$cmd

# Print the date again -- when finished
echo Finished at: `date`
