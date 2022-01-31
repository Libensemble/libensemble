## Running test run_libe_forces_funcx.py

Naive Electostatics Code Test

This is designed only as an artificial, highly configurable test
code for a libEnsemble sim func. This variant is primarily to test libEnsemble's
capability to submit simulation functions to a separate machine from where libEnsemble's
manager and workers are running.

### Forces Mini-App

A system of charged particles is set up and simulated over a number of time-steps.

Particles position and charge are initiated by a random stream.
Particles are replicated on all ranks.
**Each rank** computes forces for a subset of particles (O(N^2))
Particle force arrays are allreduced across ranks.
Particles are moved (replicated on each rank)
Total energy is appended to file forces.stat

To run forces as a standalone executable on N procs:

    mpirun -np N ./forces.x <NUM_PARTICLES> <NUM_TIMESTEPS> <SEED>

This application will need to be built on the remote machine where the sims will run.
See below.

### Running with libEnsemble.

On the remote machine:

    pip install funcx-endpoint
    funcx-endpoint configure forces

Configure the endpoint's `config.py` to include your project information and
match the machine's specifications.
See [here](https://funcx.readthedocs.io/en/latest/endpoints.html#theta-alcf) for
an example ALCF Theta configuration.

Then to run with local comms (multiprocessing) with one manager and `N` workers:

    python run_libe_forces_funcx.py --comms local --nworkers N

To run with MPI comms using one manager and `N-1` workers:

    mpirun -np N python run_libe_forces.py

Application parameters can be adjusted in `funcx_forces.yaml`.

Note that each function and path must be accessible and/or importable on the
remote machine. Absolute paths are recommended.

To remove output before the next run:

    ./cleanup.sh
