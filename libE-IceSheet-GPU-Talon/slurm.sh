#!/bin/bash -l
# 
#SBATCH --job-name=icesheet
#SBATCH --export=ALL
#SBATCH --exclusive
#
#SBATCH --time=00:30:00
#SBATCH --partition=talon-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#
#SBATCH --output=%x.%j.txt
##SBATCH --gpus-per-task=1

module list

echo "test..."
rm -r ensemble *.npy *.pickle ensemble.log lib*.txt
python -c "import libensemble"
python run_libE_on_icesheet.py --comms local --nworkers 2 
#nsys profile --stats=true mpirun -n 8 run_libe_forces.py
echo "run complete"
