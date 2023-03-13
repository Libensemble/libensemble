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

    $ conda create --name my_env python=3.8
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
from workers (see Job Submission below). The Balsam Executor can submit tasks
to the Balsam Service, which will submit these tasks dynamically to a corresponding
Balsam Site.

See the :ref:`Balsam Executor<balsam-exctr>` docs for more information.

Job Submission
--------------

On Theta, libEnsemble can be launched to two locations:

    1. **A MOM Node**: All of libEnsemble's manager and worker processes
    run centrally on a front-end MOM node. libEnsemble's MPI Executor takes
    responsibility for direct user-application submission to allocated compute nodes.
    libEnsemble must be configured to run with *multiprocessing* communications,
    since mpi4py isn't configured for use on the MOM nodes.

    1. **The Compute Nodes**: libEnsemble is submitted to Balsam, and all manager
    and worker processes are tasked to a back-end compute node and run centrally. libEnsemble's
    Balsam Executor interfaces with the Balsam service for dynamic
    user-application submission to the compute nodes.

    .. image:: ../images/balsam2.png
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
    You will need to reactivate your conda virtual environment. Configuring this
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
    export COMMS="--comms local"

    # Number of workers.
    export NWORKERS="--nworkers 128"

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
