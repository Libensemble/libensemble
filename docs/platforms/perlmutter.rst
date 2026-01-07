==========
Perlmutter
==========

Perlmutter_ is an HPE Cray "Shasta" system located at NERSC_. Its compute nodes
are equipped with four A100 NVIDIA GPUs.

It uses the SLURM scheduler to submit jobs from login nodes to run on the
compute nodes.

Configuring Python and Installation
-----------------------------------

Begin by loading the ``python`` module. The following modules are recommended::

    module load python

Create a conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create a conda_ environment in which to install libEnsemble and
all dependencies. For example::

    conda create -n libe-pm python=3.10 -y

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

    pip install libensemble

2. Install via **conda**:

.. code-block:: console

    conda config --add channels conda-forge
    conda install -c conda-forge libensemble

See :doc:`advanced installation<../advanced_installation>` for other installation options.

Job Submission
--------------

Perlmutter uses Slurm_ for job submission and management. The two most common
commands for initiating jobs are ``salloc`` and ``sbatch`` for running
in interactive and batch modes, respectively. libEnsemble runs on the compute nodes
on Perlmutter using either ``multi-processing`` (recommended) or ``mpi4py``.

While libEnsemble should detect Perlmutter settings, you can ensure this by setting
one of the following environment variables in the submission script or interactive
session for either the CPU or GPU partitions of Perlmutter:

.. code-block:: console

    export LIBE_PLATFORM="perlmutter_c"  # For CPU partition
    export LIBE_PLATFORM="perlmutter_g"  # For GPU partition

Example
-------

To run the :doc:`forces_gpu<../tutorials/forces_gpu_tutorial>` tutorial on Perlmutter.

To obtain the example you can git clone libEnsemble - although only
the forces sub-directory is needed::

    git clone https://github.com/Libensemble/libensemble
    cd libensemble/libensemble/tests/scaling_tests/forces/forces_app

To compile forces::

    module load PrgEnv-nvidia cudatoolkit craype-accel-nvidia80
    cc -DGPU -O3 -fopenmp -mp=gpu -target-accel=nvidia80 -o forces.x forces.c

Now go to forces_gpu directory::

    cd ../forces_gpu

Now grab an interactive session on one node::

    salloc -N 1 -t 20 -C gpu -q interactive -A <project_id>

Then in the session run::

    export LIBE_PLATFORM="perlmutter_g"
    python run_libe_forces.py -n 5

This places the generator on the first worker and runs simulations on the
others (each simulation using one GPU).

To see GPU usage, ssh into the node you are on in another window and run::

    watch -n 0.1 nvidia-smi

Running generator on the manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An alternative is to run the generator on a thread on the manager. The
number of workers can then be set to the number of simulation workers.

Change the ``libE_specs`` in **run_libe_forces.py** as follows.

   .. code-block:: python

    nsim_workers = ensemble.nworkers

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        gen_on_manager=True,

and run with::

    python run_libe_forces.py -n 4

To watch video
^^^^^^^^^^^^^^

    There is a video_ demonstration of the forces example on Perlmutter.

.. note::

    The video uses libEnsemble version 0.9.3, where some adjustments of the
    scripts are needed to run on Perlmutter. These adjustments are no longer
    necessary. libEnsemble now correctly detects MPI runner and GPU setting on
    Perlmutter and the GPU code runs with many more particles than the CPU
    version (forces_simple).

Example submission scripts are also given in the :doc:`examples<example_scripts>`.

Running libEnsemble with mpi4py
-------------------------------

Running libEnsemble with ``local`` comms is usually sufficient on Perlmutter. However, if you need
to use ``mpi4py``, you should install and run as follows::

    module load PrgEnv-gnu cudatoolkit
    MPICC="cc -target-accel=nvidia80 -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

This line will build ``mpi4py`` on top of a CUDA-aware Cray MPICH.

To run using 5 workers (one manager)::

    export SLURM_EXACT=1
    srun -n 6 python my_script.py

More information on using Python and ``mpi4py`` on Perlmutter can be found
in the `Python on Perlmutter`_ documentation.

Perlmutter FAQ
--------------

Some FAQs specific to Perlmutter. See more on the :doc:`FAQ<../FAQ>` page.

.. dropdown:: **srun: Job \*\*\*\*\*\* step creation temporarily disabled, retrying (Requested nodes are busy)**

   Having created a dir ``/ccs/proj/<project_id>/libensemble``:

   You may also see: ``srun: Job ****** step creation still disabled, retrying (Requested nodes are busy)``

   This error has been encountered on Perlmutter. It is recommended to add these lines to submission scripts::

       export SLURM_EXACT=1
       export SLURM_MEM_PER_NODE=0

   and to **avoid** using ``#SBATCH`` commands that may limit resources to srun job steps such as::

       #SBATCH --ntasks-per-node=4
       #SBATCH --gpus-per-task=1

   Instead provide these to sub-tasks via the ``extra_args`` option to
   the :doc:`MPIExecutor<../executor/mpi_executor>` ``submit`` function.

.. dropdown:: **GTL_DEBUG: [0] cudaHostRegister: no CUDA-capable device is detected**

   If using the environment variable ``MPICH_GPU_SUPPORT_ENABLED``, then ``srun`` commands, at
   time of writing, expect an option for allocating GPUs (e.g.~ ``--gpus-per-task=1`` would
   allocate one GPU to each MPI task of the MPI run). It is recommended that tasks submitted
   via the :doc:`MPIExecutor<../executor/mpi_executor>` specify this in the ``extra_args``
   option to the ``submit`` function (rather than using an ``#SBATCH`` command). This is needed
   even when using setting ``CUDA_VISIBLE_DEVICES`` or other options.

   If running the libEnsemble user calling script with ``srun``, then it is recommended that
   ``MPICH_GPU_SUPPORT_ENABLED`` is set in the user ``sim_f`` or ``gen_f`` function where
   GPU runs will be submitted, instead of in the batch script. E.g::

       os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "1"

.. dropdown:: **warning: /tmp/pgcudafatYDO6wtSva6K2.o: missing .note.GNU-stack section implies executable stack**

   This warning has been recently encountered when compiling the forces example
   on Perlmutter. This does not affect the run, but can be suppressed by adding
   ``-Wl,-znoexecstack`` to the build line.

Additional Information
----------------------

See the NERSC Perlmutter_ docs for more information about Perlmutter.

.. _conda: https://conda.io/en/latest/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _NERSC: https://www.nersc.gov/
.. _option to srun: https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/
.. _Python on Perlmutter: https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
.. _Slurm: https://slurm.schedmd.com/
.. _video: https://www.youtube.com/watch?v=Av8ctYph7-Y
