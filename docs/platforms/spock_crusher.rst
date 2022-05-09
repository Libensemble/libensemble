=============
Spock/Crusher
=============

Spock_ and Crusher_ are early-access testbed systems located at Oak Ridge
Leadership Computing Facility (OLCF).

Each Spock compute node consists of one 64-core AMD EPYC "Rome" CPU and four
AMD MI100 GPUs.

Each Crusher compute node contains a 64-core AMD EPYC and 4 AMD MI250X GPUs
(8 Graphics Compute Dies).

These systems use the SLURM scheduler to submit jobs from login nodes to run on the
compute nodes.

Configuring Python and Installation
-----------------------------------

Begin by loading the ``python`` module::

    module load cray-python

Job Submission
--------------

Slurm_ is used for job submission and management. libEnsemble runs on the
compute nodes using either ``multi-processing`` or ``mpi4py``.

If running more than one worker per node, the following is recommended to prevent
resource conflicts::

    export SLURM_EXACT=1
    export SLURM_MEM_PER_NODE=0

Installing libEnsemble and dependencies
---------------------------------------

libEnsemble can be installed via pip::

    pip install libensemble

Example
-------

To run the :doc:`forces_gpu<../tutorials/forces_gpu_tutorial>` tutorial on Spock or Crusher.

To obtain the example you can git clone libEnsemble - although only
the forces sub-directory is needed::

    git clone https://github.com/Libensemble/libensemble
    cd libensemble/libensemble/tests/scaling_tests/forces/forces_app

To build the forces application to use the GPU, ensure *forces.c* has the
``#pragma omp target`` line uncommented and comment out the equivalent
``#pragma omp parallel`` line.

To compile forces (in addition to cray-python module)::

    module load rocm
    module load craype-accel-amd-gfx908
    cc -I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64 -fopenmp -O3 -o forces.x forces.c

Now go to forces_gpu directory::

    cd ../forces_gpu

and modify the following lines in *forces_simf.py*. Multiply particles by 10 and change
``CUDA_VISIBLE_DEVICES`` to ``ROCR_VISIBLE_DEVICES``. So you have these modified lines::

    particles = str(int(H["x"][0][0]) * 10)
    ...
    resources.set_env_to_slots("ROCR_VISIBLE_DEVICES")

The first change will make the simulation take long enough to see.

Now grab an interactive session on one node::

    salloc --nodes=1 -A <project_id> --time=00:30:00

Then in the session run::

    export SLURM_EXACT=1  # Prevents resource conflicts on node
    python run_libe_forces.py --comms local --nworkers 4

To see GPU usage, ssh into the node you are on in another window and run::

    module load rocm
    watch -n 0.1 rocm-smi

.. _Spock:  https://docs.olcf.ornl.gov/systems/spock_quick_start_guide.html
.. _Slurm: https://slurm.schedmd.com/
.. _Crusher: https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html
