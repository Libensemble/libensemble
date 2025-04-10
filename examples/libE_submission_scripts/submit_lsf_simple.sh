#!/bin/bash -l
#BSUB -P <project code>
#BSUB -J libe_mproc
#BSUB -W 15
#BSUB -nnodes 2

python run_libe_forces.py -n 8
