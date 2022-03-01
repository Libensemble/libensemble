## Running test run_libe_forces_balsam.py

Naive Electrostatics Code Test

This is a synthetic, highly configurable simulation function. Its primary use
is to test libEnsemble's capability to submit application instances via the Balsam service,
including to separate machines from libEnsemble's processes. This means that although
this is typically a HPC scaling test, this can be run on a laptop with the `forces.x`
simulation submitted to the remote machine.

Note that this test currently requires active ALCF credentials to authenticate with
the Balsam service.

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

### Configuring Balsam

On the remote machine (in a conda or other virtual environment):

    git clone https://github.com/argonne-lcf/balsam.git
    cd balsam; pip install -e .; cd ..;
    balsam login
    balsam site init ./my-site
    cd my-site; balsam site start

You may be asked to login and authenticate with the Balsam service. Do so with
your ALCF credentials.

On any machine you've installed and logged into Balsam, you can run `balsam site ls`
to list your sites and `balsam job rm --all` to remove extraneous jobs between runs.

### Configuring and Running libEnsemble.

Configure the `RemoteForces` class in the `run_libe_forces_balsam.py` calling
script to match the Balsam site name and the path to the `forces.x` executable
on the remote machine. Configure the `submit_allocation()` function in the calling
script to correspond with the site's ID (an integer found via `balsam site ls`),
as well as the correct queue and project for the machine the Balsam site was initialized on.

Then to run with local comms (multiprocessing) with one manager and `N` workers:

    python run_libe_forces_funcx.py --comms local --nworkers N

To run with MPI comms using one manager and `N-1` workers:

    mpirun -np N python run_libe_forces.py

Application parameters can be adjusted in `funcx_forces.yaml`.

Note that each function and path must be accessible and/or importable on the
remote machine. Absolute paths are recommended.

To remove output before the next run, use:

    ./cleanup.sh

### (Optional) Configuring data-transfer via Balsam and Globus

Although the raw results of forces runs are available in Balsam sites, remote or
local, this is understandably insufficient for the simulation function's capability
to evaluate results and determine the final status of an app run if it's running
on another machine.

Balsam can coordinate data transfers via Globus between Globus endpoints. Assuming
this test is being run on a personal device, do the following to configure Globus,
then Balsam to use Globus.

- Login to [Globus](https://www.globus.org/) using ALCF or other approved organization credentials.
- Download and run [Globus Connect Personal](https://app.globus.org/file-manager/gcp) to register your device as a Globus endpoint.
- Once a Globus collection has been initialized in Globus Connect Personal, login to Globus, click "Endpoints" on the left.
- Click the collection that was created on your personal device. Copy the string after "Endpoint UUID".
- Login to the remote machine, switch to your Balsam site directory, run ``balsam site globus-login``.
- Modify ``settings.yml`` to contain a new transfer_location that matches your device, with the copied endpoint UUID.
- Run ``balsam site sync`` within the site directory to save these changes.
- Locally, in the calling script, enable ``TRANSFER_STATFILES`` and set ``GLOBUS_ENDPOINT`` to the key for the previously-defined transfer_location

This should be sufficient for ``forces.stat`` files from remote Balsam app runs
to be transferred back to your local launch directory after every app run. The
simulation function will wait for Balsam to transfer back a stat file, then determine
the calc status based on the received output.
