## Running test run_libe_forces_balsam.py

Naive Electrostatics Code Test

This is a synthetic, highly configurable simulation function. Its primary use
is to test libEnsemble's capability to submit application instances via the Balsam service,
including to separate machines from libEnsemble's processes. This means that although
this is typically an HPC scaling test, this can be run on a laptop with the `forces.x`
simulation submitted to the remote machine, and the resulting data-files transferred
back to the machine that runs the libEnsemble calling script.

Note that this test currently requires active ALCF credentials to authenticate with
the Balsam service.

### Forces Mini-App

A system of charged particles is initialized and simulated over a number of time-steps.

See `forces_app` directory for details.

**This application will need to be compiled on the remote machine**

Choose or modify a build line from `build_forces.sh` for the target platform:

    cd libensemble/libensemble/tests/scaling_tests/forces/forces_app
    ./build_forces.sh

### Configuring Balsam

On the remote machine (in a conda or other virtual environment):

    pip install balsam
    balsam login
    balsam site init ./my-site

You may be asked to login and authenticate with the Balsam service. Do so with
your ALCF credentials. Now go into the site directory:

    cd my-site

To see if the site is active, run:

    balsam site ls

If the site is not active, run:

    balsam site start

On any machine you've installed and logged into Balsam, you can run `balsam site ls`
to list your sites and `balsam job rm --all` to remove extraneous jobs between runs.

### Configuring data-transfer via Balsam and Globus

Although the raw results of forces runs are available in Balsam sites,
this is understandably insufficient for the simulation function's capability
to evaluate results and determine the final status of an app run if it's running
on another machine.

Balsam can coordinate data transfers via Globus between Globus endpoints. Assuming
this test is being run on a personal device, do the following to configure Globus,
then Balsam to use Globus.

- Login to [Globus](https://www.globus.org/) using ALCF or other approved organization credentials.
- Download and run [Globus Connect Personal](https://app.globus.org/file-manager/gcp) to register your device as a Globus endpoint. Note the initialized collection name, e.g. ``test_collection``.
- Once a Globus collection has been initialized in Globus Connect Personal, login to Globus, click "Endpoints" on the left.
- Click the collection that was created on your personal device. Copy the string after "Endpoint UUID".
- Login to the remote machine, switch to your Balsam site directory, and run ``balsam site globus-login``.
- Modify ``settings.yml`` to contain a new transfer_location that matches your device, with the copied endpoint UUID. e.g. ``test_collection: globus://19036a15-570a-12f8-bef8-22060b9b458d``
- Run ``balsam site sync`` within the site directory to save these changes.
- Locally, in the calling script (``run_libe_forces_balsam.py``), set ``GLOBUS_ENDPOINT`` to the collection name for the previously-defined transfer_location.

This should be sufficient for ``forces.stat`` files from remote Balsam app runs
to be transferred back to your personal device after every app run. The
simulation function will wait for Balsam to transfer back a stat file, then determine
the calc status based on the received output.

*To transfer files to/from Theta*, you will need to login to Globus and activate
Theta's Managed Public Endpoint:

- Login to Globus, click "Endpoints" on the left.
- Search for ``alcf#dtn_theta``, click on the result.
- On the right, click "Activate", then "Continue". Authenticate with ALCF.

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
  your home directory and/or Python paths **on the remote machine**. If running
  libEnsemble on your personal machine, feel free comment-out ``RemoteLibensembleApp.sync()``.

  **Run this script each time you edit it,** since changes to each
  ``ApplicationDefinition`` needs to be synced with the Balsam service.

2. ``run_libe_forces_balsam.py``:

  About:

  This is a typical libEnsemble plus Executor calling script, but instead of
  registering paths to apps like with the MPI Executor, this script loads the
  ``RemoteForces`` app synced with the Balsam service in ``define_apps.py``
  and registers it with libEnsemble's Balsam Executor. If running this
  script on your personal machine, it also uses the Balsam Executor to reserve
  resources at a Balsam site.

  Configuring:

  See the Globus instructions above for setting up Globus transfers within this script.

  Adjust the ``BALSAM_SITE`` field
  to match your remote Balsam site, and fields in the in the
  ``batch = exctr.submit_allocation()`` block further down. For ``site_id``,
  retrieve the corresponding field with ``balsam site ls``.

3. (optional) ``submit_libe_forces_balsam.py``:

  About:

  This Python script is effectively a batch submission script. It uses the Balsam API
  to check out resources (a ``BatchJob``) at a Balsam site, and submits libEnsemble as
  a Balsam Job onto those resources. If transferring statfiles back to your
  personal machine, it also waits until they are all returned and cancels
  the remote ``BatchJob``. *Probably only needed if running libEnsemble remotely.*

  Configuring:

  Every field in UPPER_CASE can be adjusted. ``BALSAM_SITE``, ``PROJECT``,
  and ``QUEUE`` among others will probably need adjusting. ``LIBE_NODES`` and ``LIBE_RANKS``
  specify a subset of resources specifically for libEnsemble out of ``BATCH_NUM_NODES``.

### Running libEnsemble locally

First make sure that all Balsam apps are synced with the Balsam service:

    python define_apps.py

Then run libEnsemble with multiprocessing comms, with one manager and `N` workers:

    python run_libe_forces_balsam.py --comms local --nworkers N

Or, run with MPI comms using one manager and `N-1` workers:

    mpirun -np N python run_libe_forces_balsam.py

To remove output before the next run, use:

    ./cleanup.sh

**This runs libEnsemble itself in-place, with only Forces submitted to a Balsam site.**

### (Optional) Running libEnsemble remotely

The previous instructions for running libEnsemble are understandably insufficient
if running with lots of workers or if the simulation/generation
functions are computationally expensive.

To run both libEnsemble and the Forces app on the compute nodes at Balsam site, use:

    python define_apps.py
    python submit_libe_forces_balsam.py

This routine will wait for corresponding statfiles to be transferred back from
the remote machine, then cancel the allocation.
