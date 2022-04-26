## Tutorial

This example is explained in the tutorial **Executor with Electrostatic Forces**.

https://libensemble.readthedocs.io/en/develop/tutorials/executor_forces_tutorial.html

## QuickStart

Build executable and run example. Go to `forces_app` directory and build `forces.x`:

    cd ../forces_app
    ./build_forces.sh

Then return here and run:

    python run_libe_forces.py --comms local --nworkers 4

## Running test run_libe_forces.py

Naive Electrostatics Code Test

This is a synthetic, highly configurable simulation function. Its primary use
is to test libEnsemble's capability to launch application instances via the `MPIExecutor`.

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
