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

### Using YAML in calling script (optional)

An alternative calling script `run_libe_forces_from_yaml.py` can be run in the same
way as `run_libe_forces.py` above. This uses an alternative libEnsemble interface, where
an ensemble object is created and parameters can be read from the `forces.yaml` file.

### Using batch scripts

See `examples/libE_submission_scripts`

The scripts are set up assuming a conda environment. To use the script directly
you will need to replace the following templated values if on a Cobalt system:

  <projectID> in the COBALT -A directive with your project ID.

  <conda_env_name> is the name of your conda environment.

The included scripts are.

* cobalt_submit_mproc.sh:

  Example submission script to run libEnsemble similarly on ALCF's Theta, a now-retired system.

#### Plotting Options

If either of the plotting options in the submission scripts is set to true, the scripts must be in the directory specified by PLOT_DIR. These scripts can be found in the libEnsemble project in the postproc_scripts/ directory.

        export LIBE_PLOTS=true
