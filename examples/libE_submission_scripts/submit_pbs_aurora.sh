#!/bin/bash -l
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -q <myqueue>
#PBS -A <myproject>

module load frameworks

export MPICH_GPU_SUPPORT_ENABLED=1
cd $PBS_O_WORKDIR

# 2 nodes - 12 sim workers (6 GPUs per node)
python libE_calling_script.py -n 13

# if using libE_specs["use_tiles_as_gpus"] = True
# 2 nodes 24 sim workers  (12 GPU tiles per node) libE_specs["use_tiles_as_gpus"] = True
# python libE_calling_script.py -n 25
