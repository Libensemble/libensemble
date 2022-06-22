#!/bin/bash -l
# 
#SBATCH --job-name=test
#SBATCH --export=ALL
#SBATCH --exclusive
#
#SBATCH --time=05:00:00
#SBATCH --partition=talon-gpu32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#
#SBATCH --output=test.%j.txt
##SBATCH --mail-type=end,fail
##SBATCH --mail-user=anjali.sandip@und.edu

module load shared
module load nvhpc
module list
echo ""

# uncomment the line below for single run
 ./runme.sh

# uncomment the line below for param sweep
# ./runme_syst.sh
