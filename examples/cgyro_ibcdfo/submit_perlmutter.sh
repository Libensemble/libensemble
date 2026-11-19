#!/bin/bash
#SBATCH -A m4865
# ---- full optimization: ~25-35 evaluations, 13 CGYRO runs each (also set -N 13 below)
# #SBATCH --qos=regular
# #SBATCH --time=4:00:00
# #SBATCH -N 13
# ---- smoke test: ONE evaluation, to check that 13 concurrent CGYRO runs really
#      land on 13 different nodes (also set SIMMAX=1 below)
#SBATCH --qos=debug
#SBATCH --time=0:30:00
#SBATCH -N 13
#SBATCH -J Lib_CGYRO_opt
#SBATCH -C gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmlarson@anl.gov
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Minimize G(x) = max_j gamma(x,k_j)/k_j with APOSMM + IBCDFO manifold sampling.
#
# Manifold sampling proposes ONE point at a time, so the parallelism here is over
# the k_j within an evaluation: KJ_CONCURRENCY CGYRO runs at once, one node each.
# That is why -N should equal KJ_CONCURRENCY. This differs from sample_CGYRO.py,
# where independent points are spread over libEnsemble workers instead.

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

# ---- what to optimize -------------------------------------------------------
CASE=fixed_S
# CASE=variable_S
START=0.2,0.0     # the published run started at 0.0,0.0; this is the +delta side
KYMIN=0.15        # 13 wavenumbers, matching the published run so the paths compare
# KYMIN=0.10      # 14 wavenumbers (adding k=0.10 leaves G unchanged for fixed_S)
SIMMAX=1          # smoke test: ONE evaluation. Raise to ~35 for the real run.
CACHE=/global/u2/j/jmlarson/gamma_over_KY_feb_5_S_fixed.npy
# CACHE=/global/u2/j/jmlarson/gamma_over_KY_feb_5.npy
KJ_CONCURRENCY=$SLURM_JOB_NUM_NODES
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
echo "CASE=$CASE START=$START KYMIN=$KYMIN SIMMAX=$SIMMAX KJ_CONCURRENCY=$KJ_CONCURRENCY"
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

# The cache holds every CGYRO run ever done for this case; back it up before appending.
BACKUP=${CACHE%.npy}.backup_$(date +%Y%m%d_%H%M%S).npy
cp -p "$CACHE" "$BACKUP" && echo "Backed up cache to $BACKUP"

echo "========== BEFORE PYTHON =========="
date
which python
python --version

# Fail here, clearly, rather than a minute into the allocation.
python - <<'PYEOF' || { echo "FATAL: missing Python dependencies, see above"; exit 1; }
import importlib.util
import sys
missing = [m for m in ("numpy", "jinja2", "filelock", "libensemble", "ibcdfo")
           if not importlib.util.find_spec(m)]
if missing:
    print("missing modules:", ", ".join(missing))
    print("install with:  python -m pip install --user " + " ".join(missing))
    sys.exit(1)
print("dependency check ok:", sys.executable)
PYEOF

set -x

python -u run_opt_CGYRO.py -n 1 --case $CASE --start $START \
    --ky-min $KYMIN --sim-max $SIMMAX --kj-concurrency $KJ_CONCURRENCY \
    --cache "$CACHE" \
    --cgyro-path $GACODE_ROOT/cgyro/src/cgyro \
    --wrapper $GACODE_ROOT/platform/exec/wrap.PERLMUTTER_GPU \
    --parse-cmd "python $GACODE_ROOT/cgyro/bin/cgyro_parse.py" \
    --nmpi $nmpi --nomp $nomp

EXIT_CODE=$?

set +x
echo "Python finished with exit code $EXIT_CODE"
echo "========== JOB END =========="
date
