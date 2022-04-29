## Tutorial

This example is explained in the tutorial **Executor - Assign GPUs**.

https://libensemble.readthedocs.io/en/develop/tutorials/forces_gpu_tutorial.html

Note that at time of writing the calling script `run_libe_forces.py` is identical
to `forces_simple`, and so is provided as a symlink. The `forces_simf` file has slight
modifications to assign GPUs.

## QuickStart

Go to `forces_app` directory:

    cd ../forces_app

To compile the forces application to use the GPU, ensure **forces.c** has the
`#pragma omp target` line uncommented and comment out the equivalent
`#pragma omp parallel` line. Then compile to **forces.x** using one of the GPU build
lines in build_forces.sh or similar for your platform.

    ./build_forces.sh

Then return here and run:

    python run_libe_forces.py --comms local --nworkers 4

By default, each run of forces will use one CPU and one GPU. The `forces.c` code can also
be MPI parallel and will use one GPU for each CPU rank, assuming an even split of ranks
across nodes.

## Running test run_libe_forces.py

Naive Electrostatics Code Test

This is a synthetic, highly configurable simulation function. This test aims
to show libEnsemble's capability to set assign GPU resources as needed by each
worker and to launch application instances via the `MPIExecutor`.

### Forces Mini-App

A system of charged particles is initialized and simulated over a number of time-steps.

See `forces_app` directory for details.

### Running with libEnsemble.

A random sample of seeds is taken and used as input to the sim func (forces miniapp).

In forces_app directory, modify build_forces.sh for target platform and run to
build forces.x:

    ./build_forces.sh

Then to run with local comms (multiprocessing) with one manager and `N` workers:

    python run_libe_forces.py --comms local --nworkers N

To run with MPI comms using one manager and `N-1` workers:

    mpirun -np N python run_libe_forces.py

Application parameters can be adjusted in the file `run_libe_forces.py`.

To remove output before the next run:

    ./cleanup.sh
