#!/bin/bash
#SBATCH -A m4865
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH -N 4
#SBATCH -J Lib_CGYRO
#SBATCH --gpus-per-node=4
#SBATCH -C gpu

nproc=32
nomp=8
numa=4
mpinuma=4

#source ~/.bashrc
#cd "$SLURM_SUBMIT_DIR"

. $GACODE_ROOT/shared/bin/gacode_mpi_tool

let proc_per_node=$CORES_PER_NODE/$nomp

# Run with 4 workers running simulations
export MPICH_MAX_THREAD_SAFETY=funneled
export OMP_NUM_THREADS=$nomp
export OMP_STACKSIZE=400M
export MPICH_GPU_SUPPORT_ENABLED=1

export SLURM_CPU_BIND="cores"
ulimit -c unlimited


python run_CGYRO.py -n 2  --nproc=$nproc --nomp=$nomp --numa=$numa --mpinuma=$mpinuma
