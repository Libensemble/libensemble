=======================
Summit (Decommissioned)
=======================

Summit_ was an IBM AC922 system located at the Oak Ridge Leadership Computing
Facility (OLCF). Each of the approximately 4,600 compute nodes on Summit contained two
IBM POWER9 processors and six NVIDIA Volta V100 accelerators.

Summit featured three tiers of nodes: login, launch, and compute nodes.

Users on login nodes submit batch runs to the launch nodes.
Batch scripts and interactive sessions run on the launch nodes. Only the launch
nodes can submit MPI runs to the compute nodes via ``jsrun``.

These docs are maintained to guide libEnsemble's usage on three-tier systems and/or
`jsrun` systems similar to Summit.

Configuring Python
------------------

Begin by loading the Python 3 Anaconda module::

    $ module load python

You can now create and activate your own custom conda_ environment::

    conda create --name myenv python=3.10
    export PYTHONNOUSERSITE=1 # Make sure get python from conda env
    . activate myenv

If you are installing any packages with extensions, ensure that the correct compiler
module is loaded. If using mpi4py_, this must be installed from source,
referencing the compiler. Currently, mpi4py must be built with gcc::

    module load gcc

With your environment activated, run ::

    CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

Installing libEnsemble
----------------------

Obtaining libEnsemble is now as simple as ``pip install libensemble``.
Your prompt should be similar to the following line:

.. code-block:: console

    (my_env) user@login5:~$ pip install libensemble

.. note::
    If you encounter pip errors, run ``python -m pip install --upgrade pip`` first

Or, you can install via ``conda``:

.. code-block:: console

    (my_env) user@login5:~$ conda config --add channels conda-forge
    (my_env) user@login5:~$ conda install -c conda-forge libensemble

See :doc:`here<../advanced_installation>` for more information on advanced options
for installing libEnsemble.
Special note on resource sets and Executor submit options

---------------------------------------------------------

When using the portable MPI run configuration options (e.g., num_nodes) to the
:doc:`MPIExecutor<../executor/mpi_executor>` ``submit`` function, it is important
to note that, due to the resource sets used on Summit, the options refer to
resource sets as follows:

- num_procs (int, optional) – The total number resource sets for this run.

- num_nodes (int, optional) – The number of nodes on which to submit the run.

- procs_per_node (int, optional) – The number of resource sets per node.

It is recommended that the user defines a resource set as the minimal configuration
of CPU cores/processes and GPUs. These can be added to the ``extra_args`` option
of the *submit* function. Alternatively, the portable options can be ignored and
everything expressed in ``extra_args``.

For example, the following *jsrun* line would run three resource sets,
each having one core (with one process), and one GPU, along with some extra options::

    jsrun -n 3 -a 1 -g 1 -c 1 --bind=packed:1 --smpiargs="-gpu"

To express this line in the ``submit`` function may look
something like the following::

    exctr = Executor.executor
    task = exctr.submit(app_name="mycode",
                        num_procs=3,
                        extra_args="-a 1 -g 1 -c 1 --bind=packed:1 --smpiargs="-gpu""
                        app_args="-i input")

This would be equivalent to::

    exctr = Executor.executor
    task = exctr.submit(app_name="mycode",
                        extra_args="-n 3 -a 1 -g 1 -c 1 --bind=packed:1 --smpiargs="-gpu""
                        app_args="-i input")

The libEnsemble resource manager works out the resources available to each worker,
but unlike some other systems, ``jsrun`` on Summit dynamically schedules runs to
available slots across and within nodes. It can also queue tasks. This allows variable
size runs to easily be handled on Summit. If oversubscription to the `jsrun` system
is desired, then libEnsemble's resource manager can be disabled in the
calling script via::

    libE_specs["disable_resource_manager"] = True

In the above example, the task being submitted used three GPUs, which is half those
available on a Summit node, and thus two such tasks may be allocated to each node
(from different workers), if they were running at the same time.

Job Submission
--------------

Summit used LSF_ for job management and submission. For libEnsemble, the most
important command is ``bsub`` for submitting batch scripts from the login nodes
to execute on the launch nodes.

It is recommended to run libEnsemble on the launch nodes (assuming workers are
submitting MPI applications) using the ``local`` communications mode (multiprocessing).

Interactive Runs
^^^^^^^^^^^^^^^^

You can run interactively with ``bsub`` by specifying the ``-Is`` flag,
similarly to the following::

    $ bsub -W 30 -P [project] -nnodes 8 -Is

This will place you on a launch node.

.. note::
    You will need to reactivate your conda virtual environment.

Batch Runs
^^^^^^^^^^

Batch scripts specify run settings using ``#BSUB`` statements. The following
simple example depicts configuring and launching libEnsemble to a launch node with
multiprocessing. This script also assumes the user is using the ``parse_args()``
convenience function from libEnsemble's :doc:`tools module<../utilities>`.

.. code-block:: bash

    #!/bin/bash -x
    #BSUB -P <project code>
    #BSUB -J libe_mproc
    #BSUB -W 60
    #BSUB -nnodes 128
    #BSUB -alloc_flags "smt1"

    # --- Prepare Python ---

    # Load conda module and gcc.
    module load python
    module load gcc

    # Name of conda environment
    export CONDA_ENV_NAME=my_env

    # Activate conda environment
    export PYTHONNOUSERSITE=1
    source activate $CONDA_ENV_NAME

    # --- Prepare libEnsemble ---

    # Name of calling script
    export EXE=calling_script.py

    # Communication Method
    export COMMS="--comms local"

    # Number of workers.
    export NWORKERS="--nworkers 128"

    hash -r # Check no commands hashed (pip/python...)

    # Launch libE
    python $EXE $COMMS $NWORKERS > out.txt 2>&1

With this saved as ``myscript.sh``, allocating, configuring, and queueing
libEnsemble on Summit is achieved by running ::

    $ bsub myscript.sh

Example submission scripts are also given in the :doc:`examples<example_scripts>`.

Launching User Applications from libEnsemble Workers
----------------------------------------------------

Only the launch nodes can submit MPI runs to the compute nodes via ``jsrun``.
This can be accomplished in user simulator functions directly. However, it is highly
recommended that the :doc:`Executor<../executor/ex_index>` interface
be used inside the simulator or generator, because this provides a portable interface
with many advantages including automatic resource detection, portability,
launch failure resilience, and ease of use.

.. _conda: https://conda.io/en/latest/
.. _LSF: https://www.olcf.ornl.gov/wp-content/uploads/2018/12/summit_workshop_fuson.pdf
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Summit: https://www.olcf.ornl.gov/olcf-resources/compute-systems/summit/
