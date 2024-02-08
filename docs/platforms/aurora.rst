======
Aurora
======

Aurora_ is an Intel/HPE EX supercomputer located in the ALCF_ at Argonne
National Laboratory. Each compute node contains two Intel (Sapphire Rapids)
Xeon CPUs and six Intel X\ :sup:`e` GPUs (Ponte Vecchio), each with two tiles.

The PBS scheduler is used to submit jobs from login nodes to run on compute
nodes.

Configuring Python and Installation
-----------------------------------

To obtain Python use::

    module use /soft/modulefiles
    module load frameworks

To obtain libEnsemble::

    pip install libensemble

See :doc:`here<../advanced_installation>` for more information on advanced
options for installing libEnsemble, including using Spack.

Example
-------

To run the :doc:`forces_gpu<../tutorials/forces_gpu_tutorial>` tutorial on
Aurora.

To obtain the example you can git clone libEnsemble - although only
the forces sub-directory is needed::

    git clone https://github.com/Libensemble/libensemble
    cd libensemble/libensemble/tests/scaling_tests/forces/forces_app

To compile forces (a C with OpenMP target application)::

    mpicc -DGPU -O3 -fiopenmp -fopenmp-targets=spir64 -o forces.x forces.c

Now go to forces_gpu directory::

    cd ../forces_gpu

To make use of all available GPUs, open ``run_libe_forces.py`` and adjust
the exit_criteria to do more simulations. The following will do two
simulations for each worker::

    # Instruct libEnsemble to exit after this many simulations
    ensemble.exit_criteria = ExitCriteria(sim_max=nsim_workers*2)

Now grab an interactive session on two nodes (or use the batch script at
``../submission_scripts/submit_pbs_aurora.sh``)::

    qsub -A <myproject> -l select=2 -l walltime=15:00 -lfilesystems=home -q EarlyAppAccess -I

Once in the interactive session, you may need to reload the frameworks module::

    cd $PBS_O_WORKDIR
    module use /soft/modulefiles
    module load frameworks

Then in the session run::

    python run_libe_forces.py --comms local --nworkers 13

This provides twelve workers for running simulations (one for each GPU across
two nodes). An extra worker is added to run the persistent generator. The
GPU settings for each worker simulation are printed.

Looking at ``libE_stats.txt`` will provide a summary of the runs.

Using tiles as GPUs
-------------------

If you wish to treat each tile as its own GPU, then add the *libE_specs*
option ``use_tiles_as_gpus=True``, so the *libE_specs* block of
``run_libe_forces.py`` becomes:

.. code-block:: python

    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=nsim_workers,
        sim_dirs_make=True,
        use_tiles_as_gpus=True,
    )

Now you can run again but with twice the workers for running simulations (each
will use one GPU tile)::

    python run_libe_forces.py --comms local --nworkers 25

Note that the *forces* example will automatically use the GPUs available to
each worker (with one MPI rank per GPU), so if fewer workers are provided,
more than one GPU will be used per simulation.

Also see ``forces_gpu_var_resources`` and ``forces_multi_app`` examples for
cases that use varying processor/GPU counts per simulation.

Demonstration
-------------

Note that a video demonstration_ of the *forces_gpu* example on *Frontier*
is also available. The workflow is identical when running on Aurora, with the
exception of different compiler options and numbers of workers (because the
numbers of GPUs on a node differs).

.. _ALCF: https://www.alcf.anl.gov/
.. _Aurora: https://www.alcf.anl.gov/support-center/aurorasunspot/getting-started-aurora
.. _demonstration: https://youtu.be/H2fmbZ6DnVc
