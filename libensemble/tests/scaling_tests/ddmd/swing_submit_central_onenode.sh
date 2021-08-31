#!/bin/bash -x
#SBATCH --job-name=libE-test
#SBATCH --account=STARTUP-USERNAME
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --time=00:45:00

# Make sure conda and environment are loaded and activated before sbatch

module load gcc
module load cuda/11.0.2-4szlv2t

export EXE=run_libe_ddmd.py
export NUM_WORKERS=4

export PYTHONNOUSERSITE=1

python $EXE --comms local --nworkers $NUM_WORKERS

echo The command is: $cmd
echo End PBS script information.
echo All further output is from the process being run and not the script.\n\n

$cmd

# Print the date again -- when finished
echo Finished at: `date`
