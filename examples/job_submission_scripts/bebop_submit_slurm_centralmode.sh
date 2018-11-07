#!/bin/bash
#SBATCH -J libE_test_central
#SBATCH -N 5
#SBATCH -p knlall
##SBATCH -A <my_project>
#SBATCH -o tlib.%j.%N.out 
#SBATCH -e tlib.%j.%N.error 
#SBATCH -t 01:00:00 

#Launch script for running in central mode.
#LibEnsemble will run on a dedicated node (or nodes). 
#The remaining nodes in the allocation will be dedicated to the jobs launched by the workers.

#Requirements for running:
# Must use job_controller with auto-resources=True and central_mode=True.
# Note: Requires a schedular having an environment variable giving a global nodelist in a supported format (eg. SLURM/COBALT)
#       Otherwise a worker_list file will be required.

#Currently requires even distribution - either multiple workers per node or nodes per worker


#User to edit these variables
export EXE=libE_calling_script.py
export NUM_WORKERS=4

export I_MPI_FABRICS=shm:tmi

#If using in calling script (After N mins manager kills workers and timing.dat created.)
export LIBE_WALLCLOCK=55

#---------------------------------------------------------------------------------------------
#Test
echo -e "Slurm job ID: $SLURM_JOBID"
 	
#cd $PBS_O_WORKDIR
cd $SLURM_SUBMIT_DIR

# A little useful information for the log file...
echo -e "Master process running on: $HOSTNAME"
echo -e "Directory is:  $PWD"

#This will work for the number of contexts that will fit on one node (eg. 320 on Bebop) - increase libE nodes for more.
cmd="srun --overcommit --ntasks=$(($NUM_WORKERS+1)) --nodes=1 python $EXE $LIBE_WALLCLOCK"

echo The command is: $cmd 
echo End PBS script information. 
echo All further output is from the process being run and not the pbs script.\n\n $cmd # Print the date again -- when finished 

$cmd

# Print the date again -- when finished
echo Finished at: `date`
