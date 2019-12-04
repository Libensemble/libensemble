=====
Theta
=====

Theta_ is a Cray XC40 system based on the second-generation Intel
Xeon Phi processor, available within ALCF_ at Argonne National Laboratory.

Theta features three tiers of nodes: login, MOM (Machine-Oriented Mini-server),
and compute nodes. Users on login nodes submit batch jobs to the MOM nodes.
MOM nodes execute user batch-scripts to run on the compute nodes via ``aprun``.

Theta does not allow more than one MPI application per compute node.

Configuring Python
------------------

Begin by loading the Python 3 Miniconda_ module::

    $ module load miniconda-3/latest

Create a Conda_ virtual environment, cloning the base-environment. This
environment will contain mpi4py_ and many other packages you may find useful::

    $ conda create --name my_env --clone $MINICONDA_INSTALL_PATH

.. note::
    The "Executing transaction" step of creating your new environment may take a while!

Following a successful environment-creation, the prompt will suggest activating
your new environment immediately. A Conda error may result; follow the on-screen
instructions to configure your shell with Conda.

Activate your virtual environment with::

    $ conda activate my_env

More information_ on using Conda on Theta.

Installing libEnsemble and Balsam
---------------------------------

libEnsemble
^^^^^^^^^^^

There should be an indication that your virtual environment is activated.
Obtaining libEnsemble is now as simple as ``pip install libensemble``.
Your prompt should be similar to the following line:

.. code-block:: console

    (my_env) user@thetalogin6:~$ pip install libensemble

.. note::
    If you encounter pip errors, run ``python -m pip install --upgrade pip`` first

Balsam (Optional)
^^^^^^^^^^^^^^^^^

Balsam_ is an ALCF Python utility for coordinating and executing workflows of
computations on systems like Theta. Balsam can stage in tasks to a database hosted
on a MOM node and submit these tasks dynamically to the compute nodes. libEnsemble
can also be submitted to Balsam for centralized execution on a compute-node.
libEnsemble can then submit tasks to Balsam through libEnsemble's Balsam
job-controller for execution on additional allocated nodes.

Load the Balsam module with::

    $ module load balsam/0.3.5.1

Initialize a new database similarly to the following (from the Balsam docs):

.. code-block:: bash

    $ balsam init ~/libe-workflow
    $ source balsamactivate libe-workflow
    $ balsam app --name libe-app --executable "calling.py"
    $ balsam job --name libe-job --workflow test --application libe-app --args "hello!"
    $ balsam submit-launch -A [project] -q default -t 5 -n 1 --job-mode=mpi
    $ watch balsam ls   #  follow status in realtime from command-line

Read Balsam's documentation here_.

.. note::
    Balsam will create the run directories inside the data sub-directory within the database
    directory. From here, files can be staged out to the user directory (see the example
    batch script below).

Job Submission
--------------

Theta uses Cobalt_ for job management and submission. For libEnsemble, the most
important command is ``qsub``, for submitting batch scripts from the login nodes
to execute on the MOM nodes.

On Theta, libEnsemble can be launched to two locations:

    1. **A MOM Node**: All of libEnsemble's manager and worker processes
    run on a front-end MOM node. libEnsemble's MPI job-controller takes
    responsibility for direct user-application submission to allocated compute nodes.
    libEnsemble must be configured to run with *multiprocessing* communications,
    since mpi4py isn't configured for use on the MOM nodes.

    2. **The Compute Nodes**: libEnsemble is submitted to Balsam and all manager
    and worker processes are tasked to a backend compute node. libEnsemble's
    Balsam job-controller interfaces with Balsam running on a MOM node for dynamic
    user-application submission to the compute nodes.

    .. image:: ../images/combined_ThS.png
        :alt: central_MOM
        :scale: 40
        :align: center

When considering on which nodes to run libEnsemble, consider if your user
functions execute computationally expensive code, or code built for specific
architectures. Recall also that only the MOM nodes can launch MPI jobs.

Although libEnsemble workers on the MOM nodes can technically submit
user-applications to the compute nodes directly via ``aprun`` within user functions, it
is highly recommended that the aforementioned :doc:`job_controller<../job_controller/overview>`
interface is used instead. The libEnsemble job-controller features advantages like
automatic resource-detection, portability, launch failure resilience, and ease-of-use.

Theta features one default production queue, ``default``, and two debug queues,
``debug-cache-quad`` and ``debug-flat-quad``.

.. note::
    For the default queue, the minimum number of nodes to allocate at once is 128

Module and environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure proper functioning of libEnsemble, including the ability to kill running jobs, it
recommended that the following environment variable is set::

    export PMI_NO_FORK=1

It is also recommended that the following environment modules are unloaded, if present::

    module unload trackdeps
    module unload darshan
    module unload xalt

Interactive Runs
^^^^^^^^^^^^^^^^

Users can run interactively with ``qsub`` by specifying the ``-I`` flag, similarly
to the following::

    $ qsub -A [project] -n 8 -q debug-cache-quad -t 60 -I

This will place the user on a MOM node. Then, to launch MPI jobs to the compute
nodes use ``aprun`` where you would use ``mpirun``.

.. note::
    You will need to re-activate your conda virtual environment, re-activate your
    Balsam database (if using Balsam), and reload your modules. Configuring this
    routine to occur automatically is recommended.

Batch Runs
^^^^^^^^^^

Batch scripts specify run-settings using ``#COBALT`` statements. The following
simple example depicts configuring and launching libEnsemble to a MOM node with
multiprocessing. This script also assumes the user is using the ``parse_args()``
convenience function from libEnsemble's :doc:`utils module<../utilities>`.

.. code-block:: bash

    #!/bin/bash -x
    #COBALT -t 02:00:00
    #COBALT -n 128
    #COBALT -q default
    #COBALT -A [project]
    #COBALT -O libE-project

    # --- Prepare Python ---

    # Load conda module
    module load miniconda-3/latest

    # Name of Conda environment
    export CONDA_ENV_NAME=my_env

    # Activate Conda environment
    export PYTHONNOUSERSITE=1
    source activate $CONDA_ENV_NAME

    # --- Prepare libEnsemble ---

    # Name of calling script
    export EXE=calling_script.py

    # Communication Method
    export COMMS='--comms local'

    # Number of workers.
    export NWORKERS='--nworkers 128'

    # Conda location - theta specific
    export PATH=/home/user/path/to/packages/:$PATH
    export LD_LIBRARY_PATH=/home/user/path/to/packages/:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/user/path/to/env/packages:$PYTHONPATH

    # Required for python kills on Theta
    export PMI_NO_FORK=1

    # Unload Theta modules that may interfere with job monitoring/kills
    module unload trackdeps
    module unload darshan
    module unload xalt

    python $EXE $COMMS $NWORKERS > out.txt 2>&1

With this saved as ``myscript.sh``, allocating, configuring, and queueing
libEnsemble on Theta becomes::

    $ qsub --mode script myscript.sh

Balsam Runs
^^^^^^^^^^^

Here is an example Balsam submission script:

.. code-block:: bash

    #!/bin/bash -x
    #COBALT -t 60
    #COBALT -O libE_test
    #COBALT -n 128
    #COBALT -q default
    #COBALT -A [project]

    # Name of calling script
    export EXE=calling_script.py

    # Number of workers.
    export NUM_WORKERS=128

    # Wall-clock for libE job (supplied to Balsam)
    export LIBE_WALLCLOCK=45

    # Name of working directory where Balsam places running jobs/output
    export WORKFLOW_NAME=libe_workflow

    #Tell libE manager to stop workers, dump timing.dat and exit after time.
    export SCRIPT_ARGS=$(($LIBE_WALLCLOCK-3))

    # Name of Conda environment
    export CONDA_ENV_NAME=my_env

    # Conda location - theta specific
    export PATH=/path/to/python/bin:$PATH
    export LD_LIBRARY_PATH=~/path/to/conda/env/lib:$LD_LIBRARY_PATH

    #Ensure environment isolated
    export PYTHONNOUSERSITE=1

    # Required for python kills on Theta
    export PMI_NO_FORK=1

    # Unload Theta modules that may interfere with job monitoring/kills
    module unload trackdeps
    module unload darshan
    module unload xalt

    # Activate conda environment
    . activate $CONDA_ENV_NAME

    # Activate Balsam database
    . balsamactivate default

    # Currently need at least one DB connection per worker (for postgres).
    if [[ $NUM_WORKERS -gt 128 ]]
    then
       #Add a margin
       echo -e "max_connections=$(($NUM_WORKERS+10)) #Appended by submission script" >> $BALSAM_DB_PATH/balsamdb/postgresql.conf
    fi
    wait

    # Make sure no existing apps/jobs
    balsam rm apps --all --force
    balsam rm jobs --all --force
    wait
    sleep 3

    # Add calling script to Balsam database as app and job.
    THIS_DIR=$PWD
    SCRIPT_BASENAME=${EXE%.*}

    balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

    # Running libE on one node - one manager and upto 63 workers
    balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS --wall-time-minutes $LIBE_WALLCLOCK --num-nodes 1 --ranks-per-node $((NUM_WORKERS+1)) --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" --url-in="local:/$THIS_DIR/*" --yes

    #Run job
    balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1

    . balsamdeactivate

Debugging Strategies
--------------------

View the status of your submitted jobs with ``qstat -fu [user]``.

Theta features two debug queues each with sixteen nodes. Each user can allocate
up to eight nodes at once for a maximum of one hour. Allocate nodes on a debug
queue interactively::

    $ qsub -A [project] -n 4 -q debug-flat-quad -t 60 -I

Additional Information
----------------------

See the ALCF guides_ on XC40 systems for more information about Theta.

Read the documentation for Balsam here_.

.. _ALCF: https://www.alcf.anl.gov/
.. _Theta: https://www.alcf.anl.gov/theta
.. _Balsam: https://www.alcf.anl.gov/balsam
.. _Cobalt: https://www.alcf.anl.gov/cobalt-scheduler
.. _guides: https://www.alcf.anl.gov/user-guides/computational-systems
.. _here: https://balsam.readthedocs.io/en/latest/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Conda: https://conda.io/en/latest/
.. _information: https://www.alcf.anl.gov/user-guides/conda
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
