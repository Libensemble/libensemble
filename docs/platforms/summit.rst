======
Summit
======

Summit_ is an IBM AC922 system located at the Oak Ridge Leadership Computing
Facility (OLCF). Each of the approximately 4,600 compute nodes on Summit contains two
IBM POWER9 processors and six NVIDIA Volta V100 accelerators.

Summit features three tiers of nodes: login, launch, and compute nodes.

Users on login nodes submit batch runs to the launch nodes.
Batch scripts and interactive sessions run on the launch nodes. Only the launch
nodes can submit MPI runs to the compute nodes via ``jsrun``.

Configuring Python
------------------

Begin by loading the Python 3 Anaconda module::

    $ module load python

You can now create and activate your own custom conda_ environment::

    conda create --name myenv python=3.7
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

Job Submission
--------------

Summit uses LSF_ for job management and submission. For libEnsemble, the most
important command is ``bsub`` for submitting batch scripts from the login nodes
to execute on the launch nodes.

It is recommended to run libEnsemble on the launch nodes (assuming workers are
submitting MPI applications) using the ``local`` communications mode (multiprocessing).
In the future, Balsam may be used to run libEnsemble on compute nodes.

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
convenience function from libEnsemble's :doc:`utils module<../utilities>`.

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
    export COMMS='--comms local'

    # Number of workers.
    export NWORKERS='--nworkers 128'

    hash -r # Check no commands hashed (pip/python...)

    # Launch libE
    python $EXE $COMMS $NWORKERS > out.txt 2>&1

With this saved as ``myscript.sh``, allocating, configuring, and queueing
libEnsemble on Summit is achieved by running ::

    $ bsub myscript.sh

Launching User Applications from libEnsemble Workers
----------------------------------------------------

Only the launch nodes can submit MPI runs to the compute nodes via ``jsrun``.
This can be accomplished in user ``sim_f`` functions directly. However, it is highly
recommended that the :doc:`executor<../executor/overview>` interface
be used inside the ``sim_f`` or ``gen_f``, because this provides a portable interface
with many advantages including automatic resource detection, portability,
launch failure resilience, and ease of use.

Additional Information
----------------------

See the OLCF guides_ for more information about Summit.

.. _Summit: https://www.olcf.ornl.gov/for-users/system-user-guides/summit/
.. _LSF: https://www.olcf.ornl.gov/wp-content/uploads/2018/12/summit_workshop_fuson.pdf
.. _guides: https://www.olcf.ornl.gov/for-users/system-user-guides/summit/
.. _conda: https://conda.io/en/latest/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
