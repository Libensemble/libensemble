=====
Theta
=====

Theta_ is a Cray XC40 system based on the second-generation Intel
Xeon Phi processor, available in the ALCF_ at Argonne National Laboratory.

Theta features three tiers of nodes: login, MOM,
and compute nodes. Users on login nodes submit batch jobs to the MOM nodes.
MOM nodes execute user batch scripts to run on the compute nodes via ``aprun``.

Theta will not schedule more than one MPI application per compute node.

Configuring Python
------------------

Begin by loading the Python 3 Miniconda_ module::

    $ module load miniconda-3/latest

Create a conda_ virtual environment. We recommend cloning the base
environment. This environment will contain mpi4py_ and many other packages that
are configured correctly for Theta::

    $ conda create --name my_env --clone $CONDA_PREFIX

.. note::
    The "executing transaction" step of creating your new environment may take a while!

Following a successful environment creation, the prompt will suggest activating
your new environment immediately. A conda error may result; follow the on-screen
instructions to configure your shell with conda.

Activate your virtual environment with ::

    $ export PYTHONNOUSERSITE=1
    $ conda activate my_env

Alternative
^^^^^^^^^^^

If you do not wish to clone the miniconda environment and instead create your own, and
you are using ``mpi4py`` make sure the install picks up Cray's compiler drivers. E.g::

    $ conda create --name my_env python=3.7
    $ export PYTHONNOUSERSITE=1
    $ conda activate my_env
    $ CC=cc MPICC=cc pip install mpi4py --no-binary mpi4py

More information_ on using conda on Theta is also available.

Installing libEnsemble and Balsam
---------------------------------

libEnsemble
^^^^^^^^^^^

You should get an indication that your virtual environment is activated.
Obtaining libEnsemble is now as simple as ``pip install libensemble``.
Your prompt should be similar to the following line:

.. code-block:: console

    (my_env) user@thetalogin6:~$ pip install libensemble

.. note::
    If you encounter pip errors, run ``python -m pip install --upgrade pip`` first.

Or, you can install via ``conda`` (which comes with some common dependencies):

.. code-block:: console

    (my_env) user@thetalogin6:~$ conda config --add channels conda-forge
    (my_env) user@thetalogin6:~$ conda install -c conda-forge libensemble

See :doc:`here<../advanced_installation>` for more information on advanced options
for installing libEnsemble.

Balsam (Optional)
^^^^^^^^^^^^^^^^^

Balsam_ allows libEnsemble to be run on compute nodes, and still submit tasks
from workers (see Job Submission below). The Balsam Executor will stage in tasks
to a database hosted on a MOM node, which will submit these tasks dynamically to
the compute nodes.

Balsam can be installed with::

    pip install balsam-flow

Initialize a Balsam database at a location of your choice. E.g::

    balsam init ~/myWorkflow

Further notes on using Balsam:

* Call ``balsamactivate`` in the batch script (see below). Make sure no active postgres databases are running on either login or MOM nodes before calling ``qsub``. You can check with the script ps_nodes_.

* Balsam requires PostgreSQL version 9.6.4 or later, but problems may be encountered when using the default ``pg_ctl`` and PostgreSQL 10.12 installation installed in ``/usr/bin``. This may be resolved by loading the postgresql/9.6.12 modules within submission scripts that use Balsam.

* By default there are a maximum of 128 concurrent database connections. Each worker will use a connection and a few extra are needed. Increase the number of connections by appending a new ``max_connections=`` line to ``balsamdb/postgresql.conf`` in the database directory. E.g.~ ``max_connections=1024``

* There is a Balsam module available (balsam/0.3.8), but the module's Python installation supersedes others when loaded. In practice, libEnsemble or other Python packages installed into another environment become inaccessible. Installing Balsam into a separate Python virtual environment is recommended instead.

Read Balsam's documentation here_.

.. note::
    Balsam creates run-specific directories inside ``data/my_workflow`` in the database
    directory. For example: ``$HOME/my_balsam_db/data/libe_workflow/job_run_libe_forces_b7073fa9/``.
    From here, files can be staged out (see the example batch script below).

Job Submission
--------------

On Theta, libEnsemble can be launched to two locations:

    1. **A MOM Node**: All of libEnsemble's manager and worker processes
    run centrally on a front-end MOM node. libEnsemble's MPI Executor takes
    responsibility for direct user-application submission to allocated compute nodes.
    libEnsemble must be configured to run with *multiprocessing* communications,
    since mpi4py isn't configured for use on the MOM nodes.

    2. **The Compute Nodes**: libEnsemble is submitted to Balsam, and all manager
    and worker processes are tasked to a back-end compute node and run centrally. libEnsemble's
    Balsam Executor interfaces with Balsam running on a MOM node for dynamic
    user-application submission to the compute nodes.

    .. image:: ../images/centralized_new_detailed_balsam.png
        :alt: central_Balsam
        :scale: 40
        :align: center

When considering on which nodes to run libEnsemble, consider whether your ``sim_f``
or ``gen_f`` user functions (not applications) execute computationally expensive
code, or code built specifically for the compute node architecture. Recall also
that only the MOM nodes can launch MPI applications.

Although libEnsemble workers on the MOM nodes can technically submit
user applications to the compute nodes directly via ``aprun`` within user functions, it
is highly recommended that the aforementioned :doc:`executor<../executor/overview>`
interface be used instead. The libEnsemble Executor features advantages such as
automatic resource detection, portability, launch failure resilience, and ease of use.

Theta features one default production queue, ``default``, and two debug queues,
``debug-cache-quad`` and ``debug-flat-quad``.

.. note::
    For the default queue, the minimum number of nodes to allocate at once is 128.

Module and environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to ensure proper functioning of libEnsemble, including the ability to kill running tasks,
the following environment variable should be set::

    export PMI_NO_FORK=1

It is also recommended that the following environment modules be unloaded, if present::

    module unload trackdeps
    module unload darshan
    module unload xalt

Interactive Runs
^^^^^^^^^^^^^^^^

You can run interactively with ``qsub`` by specifying the ``-I`` flag, similarly
to the following::

    $ qsub -A [project] -n 8 -q debug-cache-quad -t 60 -I

This will place you on a MOM node. Then, to launch jobs to the compute
nodes, use ``aprun`` where you would use ``mpirun``.

.. note::
    You will need to reactivate your conda virtual environment, reactivate your
    Balsam database (if using Balsam), and reload your modules. Configuring this
    routine to occur automatically is recommended.

Batch Runs
^^^^^^^^^^

Batch scripts specify run settings using ``#COBALT`` statements. The following
simple example depicts configuring and launching libEnsemble to a MOM node with
multiprocessing. This script also assumes the user is using the ``parse_args()``
convenience function from libEnsemble's :doc:`tools module<../utilities>`.

.. code-block:: bash

    #!/bin/bash -x
    #COBALT -t 02:00:00
    #COBALT -n 128
    #COBALT -q default
    #COBALT -A [project]
    #COBALT -O libE-project

    # --- Prepare Python ---

    # Obtain Conda PATH from miniconda-3/latest module
    CONDA_DIR=/soft/datascience/conda/miniconda3/latest/bin

    # Name of conda environment
    export CONDA_ENV_NAME=my_env

    # Activate conda environment
    export PYTHONNOUSERSITE=1
    source $CONDA_DIR/activate $CONDA_ENV_NAME

    # --- Prepare libEnsemble ---

    # Name of calling script
    export EXE=calling_script.py

    # Communication Method
    export COMMS='--comms local'

    # Number of workers.
    export NWORKERS='--nworkers 128'

    # Required for killing tasks from workers on Theta
    export PMI_NO_FORK=1

    # Unload Theta modules that may interfere with task monitoring/kills
    module unload trackdeps
    module unload darshan
    module unload xalt

    python $EXE $COMMS $NWORKERS > out.txt 2>&1

With this saved as ``myscript.sh``, allocating, configuring, and queueing
libEnsemble on Theta is achieved by running ::

    $ qsub --mode script myscript.sh

Balsam Runs
^^^^^^^^^^^

Here is an example Balsam submission script. It requires a pre-initialized (but not activated)
postgresql_ database. Note, the example runs libEnsemble over two dedicated nodes, reserving the
other 127 nodes for launched applications. libEnsemble is run with MPI on 128 processors
(one manager and 127 workers).:

.. code-block:: bash

    #!/bin/bash -x
    #COBALT -t 60
    #COBALT -O libE_test
    #COBALT -n 129
    #COBALT -q default
    #COBALT -A [project]

    # Name of calling script
    export EXE=calling_script.py

    # Number of workers.
    export NUM_WORKERS=127

    # Number of nodes to run libE
    export LIBE_NODES=2

    # Wall-clock for entire libE run (supplied to Balsam)
    export LIBE_WALLCLOCK=45

    # Name of working directory where Balsam places running jobs/output
    export WORKFLOW_NAME=libe_workflow

    # If user script takes ``wallclock_max`` argument.
    # export SCRIPT_ARGS=$(($LIBE_WALLCLOCK-3))
    export SCRIPT_ARGS=""

    # Name of conda environment
    export CONDA_ENV_NAME=my_env
    export BALSAM_DB_NAME=myWorkflow

    # Required for killing tasks from workers on Theta
    export PMI_NO_FORK=1

    # Unload Theta modules that may interfere with task monitoring/kills
    module unload trackdeps
    module unload darshan
    module unload xalt

    # Obtain Conda PATH from miniconda-3/latest module
    CONDA_DIR=/soft/datascience/conda/miniconda3/latest/bin

    # Ensure environment is isolated
    export PYTHONNOUSERSITE=1

    # Activate conda environment
    source $CONDA_DIR/activate $CONDA_ENV_NAME

    # Activate Balsam database
    source balsamactivate $BALSAM_DB_NAME

    # Currently need at least one DB connection per worker (for postgres).
    if [[ $NUM_WORKERS -gt 100 ]]
    then
       # Add a margin
       export BALSAM_DB_PATH=~/$BALSAM_DB_NAME  # Pre-pend with PATH
       echo -e "max_connections=$(($NUM_WORKERS+20)) # Appended by submission script" \
       >> $BALSAM_DB_PATH/balsamdb/postgresql.conf
    fi
    wait

    # Make sure no existing apps/jobs
    balsam rm apps --all --force
    balsam rm jobs --all --force
    wait
    sleep 3

    # Add calling script to Balsam database as app and job.
    export THIS_DIR=$PWD
    export SCRIPT_BASENAME=${EXE%.*}

    export LIBE_PROCS=$((NUM_WORKERS+1))  # Manager and workers
    export PROCS_PER_NODE=$((LIBE_PROCS/LIBE_NODES))  # Must divide evenly

    balsam app --name $SCRIPT_BASENAME.app --exec $EXE --desc "Run $SCRIPT_BASENAME"

    balsam job --name job_$SCRIPT_BASENAME --workflow $WORKFLOW_NAME \
    --application $SCRIPT_BASENAME.app --args $SCRIPT_ARGS \
    --wall-time-minutes $LIBE_WALLCLOCK \
    --num-nodes $LIBE_NODES --ranks-per-node $PROCS_PER_NODE \
    --url-out="local:/$THIS_DIR" --stage-out-files="*.out *.txt *.log" \
    --url-in="local:/$THIS_DIR/*" --yes

    # Run job
    balsam launcher --consume-all --job-mode=mpi --num-transition-threads=1

    wait
    source balsamdeactivate

Further examples of Balsam submission scripts can be be found in the :doc:`examples<example_scripts>`.

Debugging Strategies
--------------------

View the status of your submitted jobs with ``qstat -fu [user]``.

Theta features two debug queues each with sixteen nodes. Each user can allocate
up to eight nodes at once for a maximum of one hour. To allocate nodes on a debug
queue interactively, use ::

    $ qsub -A [project] -n 4 -q debug-flat-quad -t 60 -I

Additional Information
----------------------

See the ALCF `Support Center`_ for more information about Theta.

Read the documentation for Balsam here_.

.. _ALCF: https://www.alcf.anl.gov/
.. _Theta: https://www.alcf.anl.gov/theta
.. _Balsam: https://balsam.readthedocs.io
.. _Cobalt: https://www.alcf.anl.gov/support-center/theta/submit-job-theta
.. _`Support Center`: https://www.alcf.anl.gov/support-center/theta
.. _here: https://balsam.readthedocs.io/en/latest/
.. .. _Balsam install: https://balsam.readthedocs.io/en/latest/#quick-setup
.. _ps_nodes: https://github.com/Libensemble/libensemble/blob/develop/examples/misc/ps_nodes.sh
.. _postgresql: https://www.alcf.anl.gov/support-center/theta/postgresql-and-sqlite
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _conda: https://conda.io/en/latest/
.. _information: https://www.alcf.anl.gov/user-guides/conda
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
