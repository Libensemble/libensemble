#!/bin/bash
#SBATCH -A m4865
#SBATCH --qos=debug
#SBATCH --time=0:25:00
#SBATCH -N 1
#SBATCH -J Lib_CGYRO
#SBATCH -C gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmlarson@anl.gov

#nproc=4

nmpi=4
nomp=4
numa=1
mpinuma=4

source /global/u2/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/arash_gx_modules.sh

. $GACODE_ROOT/shared/bin/gacode_mpi_tool
let proc_per_node=$CORES_PER_NODE/$nomp
# Run with 4 workers running simulations
export MPICH_MAX_THREAD_SAFETY=funneled
export OMP_NUM_THREADS=$nomp
export OMP_STACKSIZE=400M
export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"
ulimit -c unlimited
python opt_CGYRO_3d.py -n 1 --nproc=$nmpi --nomp=$nomp --numa=$numa --mpinuma=$mpinuma
