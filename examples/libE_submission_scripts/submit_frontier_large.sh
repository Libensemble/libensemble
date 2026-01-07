#!/bin/bash
#SBATCH -J libE_warpX_full_sim_32x40
#SBATCH -A <myproject>
#SBATCH -p batch
#SBATCH --time 6:00:00
#SBATCH --nodes 240

module load cray-python

# Run one gen and 40 sim workers (6 nodes = 48 GPUs each)
python run_gpcam_warpx.py -n 41
