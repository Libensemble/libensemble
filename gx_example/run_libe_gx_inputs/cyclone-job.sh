#!/bin/bash
#SBATCH -A m4493
#SBATCH --qos=regular
#SBATCH --time=0:25:00
#SBATCH --nodes=1
#SBATCH -J gx-cyclone-hires-libe-gom
#SBATCH --constraint=gpu

# Run with 4 workers running simulations
python run_gx.py -n 4

