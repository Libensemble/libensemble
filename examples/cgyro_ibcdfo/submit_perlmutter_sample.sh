#!/bin/bash
# Perlmutter GPU job for sample_CGYRO.py: evaluate a list of points at chosen wavenumbers.
# Each point is one CGYRO run on one node, so set -N to the number of workers you want.
#
# Top up the fixed-shear cache with k=0.10 (about 205 points, ~20 node-hours):
#   sbatch -N 13 -q regular -t 04:00:00 \
#     --export=ALL,CASE=fixed_S,KY=0.10,CACHE=$SCRATCH/cgyro_cache_fixed_S.npy \
#     submit_perlmutter_sample.sh
#
# Smoke test first (4 points, debug queue):
#   sbatch -N 4 -q debug -t 00:30:00 \
#     --export=ALL,CASE=fixed_S,KY=0.10,CACHE=$SCRATCH/cgyro_cache_fixed_S.npy,MAXPTS=4 \
#     submit_perlmutter_sample.sh
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C gpu
#SBATCH -J cgyro_sample
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

MODULES_SCRIPT=/path/to/arash_gx_modules.sh
CGYRO=/global/cfs/cdirs/m4493/ebelli/gacode/cgyro/src/cgyro
WRAPPER=/global/cfs/cdirs/m4493/ebelli/gacode/platform/exec/wrap.PERLMUTTER_GPU
PARSE_CMD="python /global/cfs/cdirs/m4493/ebelli/gacode/cgyro/bin/cgyro_parse.py"

CASE=${CASE:-fixed_S}
KY=${KY:-0.10}
CACHE=${CACHE:?set CACHE to the result cache to read and extend}
XFROM=${XFROM:-$CACHE}          # by default, top up every point already in the cache
NWORKERS=${NWORKERS:-$SLURM_JOB_NUM_NODES}
NMPI=4
NOMP=4

source $MODULES_SCRIPT
. $GACODE_ROOT/shared/bin/gacode_mpi_tool

export MPICH_MAX_THREAD_SAFETY=funneled
export OMP_NUM_THREADS=$NOMP
export OMP_STACKSIZE=400M
export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"

EXTRA=""
[ -n "$MAXPTS" ] && EXTRA="--max-points $MAXPTS"

echo "nodes: $SLURM_JOB_NUM_NODES  workers: $NWORKERS  case: $CASE  k: $KY"
echo "cache: $CACHE"
date
srun -n 1 python -u sample_CGYRO.py --nworkers $NWORKERS --case $CASE \
    --cache "$CACHE" --x-from-cache "$XFROM" --ky-grid "$KY" \
    --cgyro-path $CGYRO --wrapper $WRAPPER --parse-cmd "$PARSE_CMD" \
    --nmpi $NMPI --nomp $NOMP $EXTRA
echo "exit code: $?"
date
