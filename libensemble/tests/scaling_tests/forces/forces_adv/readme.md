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

### Running with Balsam

These instructions refer to the use of Balsam before v0.6. This now uses the
Legacy Balsam executor in libEnsemble.

This Balsam does not support multi-site (see balsam_forces directory for multi-site Balsam forces).

This version is only recommended if that one cannot be accessed on the system or does not support
a required feature.

To run with balsam, set `USE_BALSAM = True` in `run_libe_forces.py`.
You need to have followed the instructions to install `balsam` and set-up/activate a database.
(See https://github.com/argonne-lcf/balsam).

Then to test locally, run the `balsam_local.sh` script. The default runs with 2 workers.

    ./balsam_local.sh

The running tasks can be seen inside the balsam database dir `<DIR>/data/libe_workflow/`.
While the key output files will be copied back to the run dir at completion. Also see
the log in `<DIR>/log` if there are any issues. To run on batch systems, see the example
scripts such as `theta_submit_balsam.sh`.

### Using batch scripts

The scripts are set up assuming a conda environment. To use the script directly
you will need to replace the following templated values:

  <projectID> in the COBALT -A directive with your project ID.

  <conda_env_name> is the name of your conda environment.

and in Balsam scripts:

  <dbase_name> The name of an initialized balsam database.
               (with max_connections enough for the number of workers)

The included scripts are.

* theta_submit_mproc.sh:

  Example Theta submission script to run libEnsemble in central mode on the Theta launch (MOM) nodes with multiprocessing worker concurrency.

* summit_submit_mproc.sh:

  Example Summit submission script to run libEnsemble in central mode on the Summit launch nodes with multiprocessing worker concurrency.

* theta_submit_balsam.sh:

  Example Theta submission script to run libEnsemble in central mode with MPI worker concurrency using Balsam. In this case, the libEnsemble manager and workers run on compute nodes and submit jobs via Balsam.

#### Plotting Options

If either of the plotting options in the submission scripts is set to true, the scripts must be in the directory specified by PLOT_DIR. These scripts can be found in the libEnsemble project in the postproc_scripts/ directory.

        export LIBE_PLOTS=true
        export BALSAM_PLOTS=true

#### Note on theta_submit_balsam.sh

Adjusting the node/core/worker count.: The NUM_WORKERS variable is only
currently used if libEnsemble is running on one node, in which case it should
be one less than the number of nodes in the job allocation (leaving one
dedicated node to run libEnsemble). If more workers are used then the variables
NUM_NODES and RANKS_PER_NODE need to be explicitly set (these are for
libEnsemble which will require one task for the manager and the rest will be
workers). The total node allocation (in the COBALT -n directive) will need to
be the number of nodes for libEnsemble plus the number of nodes for each worker to
launch jobs to.
