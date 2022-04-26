======================
libEnsemble with SLURM
======================

SLURM is a popular open-source workload manager.

libEnsemble is able to read SLURM node lists and partition these to workers. By
default this is done by :ref:`reading an environment variable<resource_detection>`.

Example SLURM submission scripts for various systems are given in the
:doc:`examples<example_scripts>`. Further examples are given in some of the specific
platform guides (e.g. :doc:`Perlmutter guide<perlmutter>`)

By default, the :doc:`MPIExecutor<../executor/mpi_executor>` uses ``mpirun``
as a priority over ``srun`` as it works better in some cases. If ``mpirun`` does
not work well, then try telling the MPIExecutor to use ``srun`` when it is initiated
in the calling script::

    from libensemble.executors.mpi_executor import MPIExecutor
    exctr = MPIExecutor(custom_info={'mpi_runner':'srun'})

Common Errors
-------------

SLURM systems can have various configurations which may affect what is required
when assigning more than one worker to any given node.

**srun: Job \*\*\*\*\*\* step creation temporarily disabled, retrying (Requested nodes are busy)**

You may also see: ``srun: Job ****** step creation still disabled, retrying (Requested nodes are busy)``

It is recommended to add these to submission scripts to prevent resource conflicts::

    export SLURM_EXACT=1
    export SLURM_MEM_PER_NODE=0

Alternatively, the ``--exact`` `option to srun`_, along with other relevant options
can be given on any ``srun`` lines (including the ``MPIExecutor`` submission lines
via the ``extra_args`` option).

Secondly, while many configurations are possible, it is recommended to **avoid** using
``#SBATCH`` commands that may limit resources to srun job steps such as::

    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus-per-task=1

Instead provide these to sub-tasks via the ``extra_args`` option to the
:doc:`MPIExecutor<../executor/mpi_executor>` ``submit`` function.

**GTL_DEBUG: [0] cudaHostRegister: no CUDA-capable device is detected**

If using the environment variable ``MPICH_GPU_SUPPORT_ENABLED``, then ``srun`` commands may
expect an  option for allocating GPUs (e.g.~ ``--gpus-per-task=1`` would
allocate one GPU to each MPI task of the MPI run). It is recommended that tasks submitted
via the :doc:`MPIExecutor<../executor/mpi_executor>` specify this in the ``extra_args``
option to the ``submit`` function (rather than using an ``#SBATCH`` command). This is needed
even when using setting ``CUDA_VISIBLE_DEVICES`` or other options.

If running the libEnsemble user calling script with ``srun``, then it is recommended that
``MPICH_GPU_SUPPORT_ENABLED`` is set in the user ``sim_f`` or ``gen_f`` function where
GPU runs will be submitted, instead of in the batch script. E.g::

    os.environ['MPICH_GPU_SUPPORT_ENABLED'] = "1"

Note on Resource Binding
------------------------

Note that the use of ``CUDA_VISIBLE_DEVICES`` and other environment variables is often
a highly portable way of assigning specific GPUs to workers, and has been known to work
on some systems when other methods do not. See the libEnsemble regression test `test_persistent_sampling_CUDA_variable_resources.py`_ for an example of setting
CUDA_VISIBLE_DEVICES in the imported simulator function (``six_hump_camel_CUDA_variable_resources``).

On other systems, like Perlmutter, using an option such as ``--gpus-per-task=1`` or
``-gres=gpu:1`` in ``extra_args`` is sufficient to allow SLURM to find the free GPUs.

Note that the ``srun`` options such as::

    --gpu-bind=map_gpu:2,3

do not necessarily provide absolute GPU slots when there are more than one concurrent
job steps (``sruns``) running on a node. If desired, such options could be set using the
:doc:`worker resources<../resource_manager/worker_resources>` module in a similar manner
to how ``CUDA_VISIBLE_DEVICES`` is set in the example.

Some useful commands
--------------------

Find slurm version::

    scontrol --version

Find SLURM system configuration::

    scontrol show config

Find SLURM partition configuration for a partition called 'gpu'::

    scontrol show partition gpu

.. _option to srun: https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel
.. _test_persistent_sampling_CUDA_variable_resources.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_sampling_CUDA_variable_resources.py
