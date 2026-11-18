#!/bin/bash
# Perlmutter GPU job for run_opt_CGYRO.py.  Edit the account and the three paths, and set -N to the number of
# CGYRO runs you want executing at once (one node each); KJ_CONCURRENCY must not exceed it.
#
# Smoke test:   sbatch -N 4  --export=ALL,CASE=fixed_S,SIMMAX=1,KJ_CONCURRENCY=4  -q debug -t 00:30:00 submit_perlmutter.sh
# Full eval:    sbatch -N 13 --export=ALL,CASE=fixed_S,SIMMAX=36,KJ_CONCURRENCY=13 -q regular -t 04:00:00 submit_perlmutter.sh
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C gpu
#SBATCH -J cgyro_ibcdfo
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

MODULES_SCRIPT=/path/to/arash_gx_modules.sh
CGYRO=/global/cfs/cdirs/m4493/ebelli/gacode/cgyro/src/cgyro
WRAPPER=/global/cfs/cdirs/m4493/ebelli/gacode/platform/exec/wrap.PERLMUTTER_GPU
PARSE_CMD="python /global/cfs/cdirs/m4493/ebelli/gacode/cgyro/bin/cgyro_parse.py"

CASE=${CASE:-fixed_S}
SIMMAX=${SIMMAX:-36}
KJ_CONCURRENCY=${KJ_CONCURRENCY:-1}
CACHE=${CACHE:-$SCRATCH/cgyro_cache_${CASE}.npy}   # reuse the same file to restart a run
NMPI=4
NOMP=4

source $MODULES_SCRIPT
. $GACODE_ROOT/shared/bin/gacode_mpi_tool

export MPICH_MAX_THREAD_SAFETY=funneled
export OMP_NUM_THREADS=$NOMP
export OMP_STACKSIZE=400M
export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"

echo "nodes: $SLURM_JOB_NUM_NODES  concurrency: $KJ_CONCURRENCY  case: $CASE  sim_max: $SIMMAX  cache: $CACHE"
date
python -u run_opt_CGYRO.py -n 1 --case $CASE --sim-max $SIMMAX --kj-concurrency $KJ_CONCURRENCY \
    --cache $CACHE --cgyro-path $CGYRO --wrapper $WRAPPER --parse-cmd "$PARSE_CMD" --nmpi $NMPI --nomp $NOMP
echo "exit code: $?"
date
