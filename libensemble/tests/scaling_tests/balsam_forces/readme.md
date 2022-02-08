## Running test run_libe_forces_balsam.py

Naive Electrostatics Code Test

This is a synthetic, highly configurable simulation function. Its primary use
is to test libEnsemble's capability to submit application instances via the Balsam service,
including to separate machines from libEnsemble's processes.

### Forces Mini-App

A system of charged particles is initialized and simulated over a number of time-steps.

Particles' position and charge are initiated using a random stream.
Particles are replicated on all ranks.
**Each rank** computes forces for a subset of particles (`O(N^2)` operations).
Particle force arrays are `allreduced` across ranks.
Particles are moved (replicated on each rank).
Total energy is appended to the forces.stat file.

To run forces as a standalone executable on `N` procs:

    mpirun -np N ./forces.x <NUM_PARTICLES> <NUM_TIMESTEPS> <SEED>

This application will need to be compiled on the remote machine where the sim_f will run.
See below.

### Running with libEnsemble.

On the remote machine:

    git clone https://github.com/argonne-lcf/balsam.git
    cd balsam; pip install -e .
    cd ..; balsam site init ./my-site

You may be asked to login and authenticate with the Balsam service. Do so with
your ALCF credentials.

Configure the `RemoteForces` class in the `run_libe_forces_balsam.py` calling
script to match the Balsam site name and the path to your `forces.x` executable.
Configure the path to the Balsam site's `data` directory in `balsam_forces.yaml`
to match the path to your site's corresponding directory. Configure the
`submit_allocation()` function in the calling script to correspond with your site's
ID (an integer found via `balsam site ls`), as well as the correct queue and project
for the machine the Balsam site was initialized on.

Then to run with local comms (multiprocessing) with one manager and `N` workers:

    python run_libe_forces_funcx.py --comms local --nworkers N

To run with MPI comms using one manager and `N-1` workers:

    mpirun -np N python run_libe_forces.py

Application parameters can be adjusted in `funcx_forces.yaml`.

Note that each function and path must be accessible and/or importable on the
remote machine. Absolute paths are recommended.

To remove output before the next run, use:

    ./cleanup.sh
