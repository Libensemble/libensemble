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

### Configuring libEnsemble

There are several scripts that each need to be adjusted. To explain each:

1. ``define_apps.py``:

  About:

  This script defines and syncs each of our Balsam apps with the Balsam service. A Balsam
  app is an ``ApplicationDefinition`` class with ``site`` and
  ``command_template`` fields. ``site`` specifies to Balsam on which Balsam site
  the app should be run, and ``command_template`` specifies the command (as a Jinja2
  string template) that should be executed. This script contains two apps, ``RemoteLibensembleApp``
   and ``RemoteForces``. If you're running libEnsemble on your personal machine and
   only submitting the Forces app via Balsam, only ``RemoteForces`` needs adjusting.

  Configuring:

  Adjust the ``site`` field in each ``ApplicationDefinition`` to match your remote
  Balsam site. Adjust the various paths in the ``command_template`` fields to match
  your home directory and/or Python paths **on the remote machine**.

  **Run this script each time you edit it,** since changes to each
  ``ApplicationDefinition`` need to be synced with the Balsam service.

2. ``run_libe_forces_balsam.py``:

  About:

  This is a typical libEnsemble plus Executor calling script, but instead of
  registering paths to apps as with the MPI Executor, this script loads the
  ``RemoteForces`` app synced with the Balsam service in ``define_apps.py``
  and registers it with libEnsemble's Balsam Executor. If running this
  script on your personal machine, it also uses the Balsam Executor to check
  out resources at a Balsam site.

  Configuring:

  At a minimum (if not transferring statfiles), adjust the ``BALSAM_SITE`` field
  to match your remote Balsam site, and fields in the in the
  ``batch = exctr.submit_allocation()`` block further down. For ``site_id``,
  retrieve the corresponding field with ``balsam site ls``. If this script is being
  run on a remote machine, the ``forces.from_yaml()`` path can be adjusted to point to
  the ``balsam_forces.yaml`` configuration file on that machine so it doesn't have
  to be transferred over.

3. (optional) ``submit_libe_forces_balsam.py``:

  About:

  This Python script is effectively a batch submission script. It uses the Balsam API
  to check out resources at a Balsam site, and submits libEnsemble as
  a Balsam Job onto those resources. Note that customizing the Globus transfer
  of the ``balsam_forces.yaml`` file is necessary

  Configuring:


### Running libEnsemble

Then to run with local comms (multiprocessing) with one manager and `N` workers:

    python run_libe_forces_balsam.py --comms local --nworkers N

To run with MPI comms using one manager and `N-1` workers:

    mpirun -np N python run_libe_forces.py

**This run libEnsemble itself in-place, with only Forces submitted to a Balsam site.**

To run both libEnsemble and the Forces app on a Balsam site, use:

  python submit_libe_forces_balsam.py

Application parameters can be adjusted in `balsam_forces.yaml`.

Note that each function and path must be accessible and/or importable on the
remote machine. Absolute paths are recommended.
**This runs libEnsemble itself in-place, with only forces submitted to a Balsam site.**

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

*To transfer files to Theta*, you will need to login to Globus and activate
the ``alcf#dtn_theta`` Managed Public Endpoint.

### (Optional) Running libEnsemble as a Balsam app on compute nodes

The previous instructions for running libEnsemble are understandably insufficient
if running with potentially hundreds of workers or if the simulation/generation
functions are computationally expensive.

The included ``submit_libe_forces_balsam.py`` script will submit libEnsemble itself
as a Balsam Job, to be run by a Balsam site on the compute nodes. From there libEnsemble's
simulation function will behave as before, submitting forces apps to Balsam for scheduling
on the same allocation.

Since Balsam's API can initiate allocations for a given Balsam site remotely,
``submit_libe_forces_balsam.py`` behaves like a batch submission script except
it can be run from *anywhere* and still initiate a session on Theta. This does mean
that any input files still need to be transferred by Globus to be accessible by
libEnsemble running on the compute nodes. Customize the ``input_file`` dictionary
according to Balsam's Globus specifications to do this (see the previous section).

The following parameters can be adjusted at the top of this script:

    SIM_MAX = 16  # make sure matches in balsam_forces.yaml
    BATCH_NUM_NODES = 5
    BATCH_WALL_CLOCK_TIME = 60
    PROJECT = "CSC250STMS07"
    QUEUE = "debug-flat-quad"

    # libE Job Parameters - Will use above resources
    LIBE_NODES = 1
    LIBE_RANKS = 5

**Adjust each of the literal sites, directories, paths and other attributes**
in each of the ``ApplicationDefinition`` instances. If transferring statfiles,
this script can wait for a number of statfiles equal to ``sim_max`` to be returned,
then cancel the remote BatchJob. For this script, set ``TRANSFER_STATFILES`` to ``True.``
The calling script will also need to be updated to contain the correct Globus endpoint
and destination directory for the transfers.
