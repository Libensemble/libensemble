#!/bin/bash -l
# 
#SBATCH --job-name=test
#SBATCH --account=libe_gpu
#SBATCH --export=ALL
#SBATCH --exclusive
#
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#
#SBATCH --output=test.%j.txt
##SBATCH --mail-type=end,fail
##SBATCH --mail-user=wangk@anl.gov

module load shared
module load nvhpc
module list
echo ""

# uncomment the line below for single run
nvcc -arch=sm_70 -O3 ssa_fem_pt.cu

# run the code
nsys profile --stats=true ./a.out

#./runme.sh

# uncomment the line below for param sweep
# ./runme_syst.sh
                 
