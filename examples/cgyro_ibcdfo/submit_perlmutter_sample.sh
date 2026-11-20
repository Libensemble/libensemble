#!/bin/bash
#SBATCH -A m4865
# ---- debug queue: at most 8 nodes and 30 minutes. At ~5 min per k=0.10 run that
#      is ~5 waves, so ~40 points per job; the 249-point variable-shear top-up
#      therefore needs ~6 submissions. Re-running is free: points already in the
#      cache are skipped, so just resubmit until it reports "0 need work".
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH -N 8
# ---- regular queue: the whole top-up in one go
# #SBATCH --qos=regular
# #SBATCH --time=3:00:00
# #SBATCH -N 13
#SBATCH -J Lib_CGYRO
#SBATCH -C gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmlarson@anl.gov
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Evaluate CGYRO at a fixed list of points for one wavenumber, to top up an
# existing result cache. One point per worker, one node per point, so the
# number of workers is the number of nodes requested above.

echo "========== JOB START =========="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================"

nmpi=4
nomp=4
numa=1
mpinuma=4

# ---- what to evaluate -------------------------------------------------------
CASE=variable_S
# CASE=fixed_S
KY=0.10
CACHE=/global/u2/j/jmlarson/gamma_over_KY_feb_5.npy
# CACHE=/global/u2/j/jmlarson/gamma_over_KY_feb_5_S_fixed.npy
# MAXPTS=4        # uncomment to limit the run to this many points (smoke test)
NWORKERS=$SLURM_JOB_NUM_NODES
# -----------------------------------------------------------------------------

echo "Loading modules..."
source /global/u2/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/arash_gx_modules.sh
echo "Modules loaded."

echo "Setting up GACODE MPI tool..."
. $GACODE_ROOT/shared/bin/gacode_mpi_tool
echo "GACODE setup complete."

let proc_per_node=$CORES_PER_NODE/$nomp

echo "Derived settings:"
echo "CORES_PER_NODE=$CORES_PER_NODE"
echo "proc_per_node=$proc_per_node"
echo "nmpi=$nmpi nomp=$nomp numa=$numa mpinuma=$mpinuma"
echo "CASE=$CASE KY=$KY NWORKERS=$NWORKERS MAXPTS=${MAXPTS:-all}"
echo "CACHE=$CACHE"

# Environment variables
export MPICH_MAX_THREAD_SAFETY=funneled
export OMP_NUM_THREADS=$nomp
export OMP_STACKSIZE=400M
export MPICH_GPU_SUPPORT_ENABLED=1
export SLURM_CPU_BIND="cores"

echo "Environment variables set:"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_STACKSIZE=$OMP_STACKSIZE"
echo "MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED"

ulimit -c unlimited

# The cache is the only copy of every CGYRO run ever done for this case, so keep
# a backup before appending to it. Runs only add entries, never change them.
BACKUP=${CACHE%.npy}.backup_$(date +%Y%m%d_%H%M%S).npy
cp -p "$CACHE" "$BACKUP" && echo "Backed up cache to $BACKUP"

EXTRA=""
[ -n "$MAXPTS" ] && EXTRA="--max-points $MAXPTS"

echo "========== BEFORE PYTHON =========="
date
which python
python --version
set -x

python -u sample_CGYRO.py -n $NWORKERS --case $CASE \
    --cache "$CACHE" --x-from-cache "$CACHE" --ky-grid $KY \
    --cgyro-path $GACODE_ROOT/cgyro/src/cgyro \
    --wrapper $GACODE_ROOT/platform/exec/wrap.PERLMUTTER_GPU \
    --parse-cmd "python $GACODE_ROOT/cgyro/bin/cgyro_parse.py" \
    --nmpi $nmpi --nomp $nomp $EXTRA

EXIT_CODE=$?

set +x
echo "Python finished with exit code $EXIT_CODE"
echo "========== JOB END =========="
date
