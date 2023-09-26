Test to compare overheads associated with on-node concurrency
-------------------------------------------------------------

Runs batches (by default of 4 runs each) of single proc/gpu forces. A batch of
runs is concurrent using the GPUs on a node. Once one batch is complete another
one runs.

The standard timing study does two sets of two batches, and then two sets of
four batches and produces a file time.out inside the output directory (which
is the platform name appended by datetime).

Be sure to modify the project code in the submission script as required.

Instructions for Perlmutter
---------------------------

module load PrgEnv-nvidia cudatoolkit craype-accel-nvidia80

cc -DGPU -Wl,-znoexecstack -O3 -fopenmp -mp=gpu -target-accel=nvidia80 -o forces.x forces.c

# Modify submit_perlmutter.sh for project ID and submit
sbatch submit_perlmutter.sh


Instructions for Frontier
-------------------------

module load rocm craype-accel-amd-gfx90a

cc -DGPU -I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64 -fopenmp -O3 -o forces.x forces.c

# Modify submit_frontier.sh for project ID and submit
sbatch submit_frontier.sh
