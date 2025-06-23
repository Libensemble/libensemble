#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --account=m4505
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --job-name=test-w7x-gx
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmlarson@anl.gov

pushd .
cd ../gx
source ../gx.config
popd

source ../t3d/.venv/bin/activate

export NCCL_DEBUG=WARN

python run_libe.py
