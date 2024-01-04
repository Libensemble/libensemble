#!/bin/bash -l
#PBS -l select=2
#PBS -l walltime=00:15:00
#PBS -q <myqueue>
#PBS -A <myproject>

module use /soft/modulefiles
module load frameworks

export MPICH_GPU_SUPPORT_ENABLED=1
cd $PBS_O_WORKDIR

# 2 nodes - 12 sim workers (6 GPUs per node)
python run_libe_forces.py --comms local --nworkers 13

# if using libE_specs["use_tiles_as_gpus"] = True
# 2 nodes 24 sim workers  (12 GPU tiles per node)
# python run_libe_forces.py --comms local --nworkers 25
