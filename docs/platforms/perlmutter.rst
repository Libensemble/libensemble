==========
Perlmutter
==========

Perlmutter_ is an HPE Cray “Shasta” system located at NERSC_.
Its compute nodes are equipped with four A100 NVIDIA GPUs.
It uses the SLURM scheduler to submit jobs from login nodes to run on the
compute nodes.

Configuring Python and Installation
-----------------------------------

Begin by loading the ``python`` module. The following modules are recommended::

    module load PrgEnv-gnu cudatoolkit python

Create a conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create a conda_ environment in which to install libEnsemble and
all dependencies. For example::

    conda create -n libe-pm python=3.9 -y

As Perlmutter has a shared HOME filesystem with other clusters, using
the ``-pm`` suffix (for Perlmutter) is good practice.

Activate your virtual environment with::

    export PYTHONNOUSERSITE=1
    conda activate libe-pm

Installing libEnsemble and dependencies
---------------------------------------

Having loaded the Anaconda Python module, libEnsemble can be installed
by one of the following ways.

1. Install via **pip** into the environment.

.. code-block:: console

    (my_env) user@cori07:~$ pip install libensemble

2. Install via **conda**:

.. code-block:: console

    (my_env) user@cori07:~$ conda config --add channels conda-forge
    (my_env) user@cori07:~$ conda install -c conda-forge libensemble

See :doc:`here<../advanced_installation>` for more information on advanced options
for installing libEnsemble, including using Spack.

Installing mpi4py
^^^^^^^^^^^^^^^^^

If using ``mpi4py`` for communications (optional), it is recommended that you install
using the following line (having installed the ``cudatoolkit`` module)::

    MPICC="cc -target-accel=nvidia80 -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

This line will build ``mpi4py`` on top of a CUDA-aware Cray MPICH.

More information on using Python and ``mpi4py`` on Perlmutter can be found
in the `Python on Perlmutter`_ documentation.

Job Submission
--------------

Perlmutter uses Slurm_ for job submission and management. The two most common
commands for initiating jobs are ``salloc`` and ``sbatch`` for running
in interactive and batch modes, respectively. libEnsemble runs on the compute nodes
on Perlmutter using either ``multi-processing`` or ``mpi4py``.

If running more than one worker per node, the following is recommended to prevent
resource conflicts::

    export SLURM_EXACT=1
    export SLURM_MEM_PER_NODE=0

Alternatively, the ``--exact`` `option to srun`_, along with other relevant options
can be given on any ``srun`` lines (including executor submission lines via the
``extra_args`` option).

Example
-------

A simple example batch script for a libEnsemble use case that runs 5 workers (one
generator and four simulators) on one node:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH -J libE_small_test
    #SBATCH -A <myproject_g>
    #SBATCH -C gpu
    #SBATCH --time 15
    #SBATCH --nodes 1

    export MPICH_GPU_SUPPORT_ENABLED=1
    export SLURM_EXACT=1
    export SLURM_MEM_PER_NODE=0

    python libe_calling_script.py --comms local --nworkers 5

.. note::
    Any loaded modules and environment variables (including conda environments) are
    inherited by the job on Perlmutter.

This example calling script has the following line so the node is divided into
four resource sets (the example generator does not need dedicated resources):

.. code-block:: python

    libE_specs['num_resource_sets'] = 4

The MPIExecutor is also initiated in the calling script, ensuring that ``srun`` is picked up::

    from libensemble.executors.mpi_executor import MPIExecutor
    exctr = MPIExecutor(custom_info={'mpi_runner':'srun'})

Each worker runs a simulator function that uses the :doc:`MPIExecutor<../executor/mpi_executor>`
``submit`` function, including the argument ``--gpus-per-task=1``::

    from libensemble.executors.executor import Executor
    exctr = Executor.executor
    task = exctr.submit(app_name='sim1',
                        num_procs=n_rsets,
                        app_args='input.txt',
                        extra_args='--gpus-per-task=1'
                        )

If running using :doc:`variable resource workers<../resource_manager/overview>`,
between one and four-way MPI runs may be issued by any of the workers (with each
MPI task using a GPU). libEnsemble's resource manager automatically disables workers
whose resources are being used by another worker.

Example submission scripts are also given in the :doc:`examples<example_scripts>`.

Perlmutter FAQ
--------------

**srun: Job \*\*\*\*\*\* step creation temporarily disabled, retrying (Requested nodes are busy)**

You may also see: ``srun: Job ****** step creation still disabled, retrying (Requested nodes are busy)``

This error has been encountered on Perlmutter. It is recommended to add these to submission scripts::

    export SLURM_EXACT=1
    export SLURM_MEM_PER_NODE=0

and to **avoid** using ``#SBATCH`` commands that may limit resources to srun job steps such as::

    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus-per-task=1

Instead provide these to sub-tasks via the ``extra_args`` option to
the :doc:`MPIExecutor<../executor/mpi_executor>` ``submit`` function.

**GTL_DEBUG: [0] cudaHostRegister: no CUDA-capable device is detected**

If using the environment variable ``MPICH_GPU_SUPPORT_ENABLED``, then ``srun`` commands, at
time of writing, expect an  option for allocating GPUs (e.g.~ ``--gpus-per-task=1`` would
allocate one GPU to each MPI task of the MPI run). It is recommended that tasks submitted
via the :doc:`MPIExecutor<../executor/mpi_executor>` specify this in the ``extra_args``
option to the ``submit`` function (rather than using an ``#SBATCH`` command). This is needed
even when using setting ``CUDA_VISIBLE_DEVICES`` or other options.

If running the libEnsemble user calling script with ``srun``, then it is recommended that
``MPICH_GPU_SUPPORT_ENABLED`` is set in the user ``sim_f`` or ``gen_f`` function where
GPU runs will be submitted, instead of in the batch script. E.g::

    os.environ['MPICH_GPU_SUPPORT_ENABLED'] = "1"

Additional Information
----------------------

See the NERSC Perlmutter_ docs for more information about Perlmutter.

.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/
.. _Python on Perlmutter: https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
.. _option to srun: https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel
.. _conda: https://conda.io/en/latest/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Slurm: https://slurm.schedmd.com/
.. _NERSC: https://www.nersc.gov/
