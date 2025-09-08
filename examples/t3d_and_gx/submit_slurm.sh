#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --account=m4505
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --job-name=reg-w7x-gx
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmlarson@anl.gov

echo $SLURM_JOB_NODELIST > nodes.$SLURM_JOB_ID

pushd .
cd ~/jai/gx
source ../gx.config
popd

source /global/u2/j/jmlarson/jai/t3d/.venv/bin/activate

export NCCL_DEBUG=WARN

date > time.$SLURM_JOB_ID
python run_libe.py
date >> time.$SLURM_JOB_ID
