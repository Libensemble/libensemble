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

To obtain Python and create a virtual environment:

.. code-block:: console

    module load frameworks
    python -m venv /path/to-venv --system-site-packages
    . /path/to-venv/bin/activate

where ``/path/to-venv`` can be anywhere you have write access. For future sessions,
just load the frameworks module and run the activate line.

To obtain libEnsemble::

    pip install libensemble

See :doc:`here<../advanced_installation>` for more information on advanced
options for installing libEnsemble, including using Spack.

Example
-------

To run the :doc:`forces_gpu<../tutorials/forces_gpu_tutorial>` tutorial on
Aurora.

To obtain the example you can git clone libEnsemble - although only
the ``forces`` sub-directory is strictly needed::

    git clone https://github.com/Libensemble/libensemble
    cd libensemble/libensemble/tests/scaling_tests/forces/forces_app

To compile forces (a C with OpenMP target application)::

    mpicc -DGPU -O3 -fiopenmp -fopenmp-targets=spir64 -o forces.x forces.c

Now go to forces_gpu directory::

    cd ../forces_gpu

To make use of all available GPUs, open **run_libe_forces.py** and adjust
the ``exit_criteria`` to perform more simulations. The following will run two
simulations for each worker:

.. code-block:: python

    # Instruct libEnsemble to exit after this many simulations
    ensemble.exit_criteria = ExitCriteria(sim_max=nsim_workers*2)

Now grab an interactive session on two nodes (or use the batch script at
``../submission_scripts/submit_pbs_aurora.sh``)::

    qsub -A <myproject> -l select=2 -l walltime=15:00 -lfilesystems=home:flare -q debug -I

Once in the interactive session, you may need to reload the frameworks module::

    cd $PBS_O_WORKDIR
    . /path/to-venv/bin/activate

Then in the session run::

    python run_libe_forces.py -n 13

This provides twelve workers for running simulations (one for each GPU across
two nodes). An extra worker is added to run the persistent generator. The
GPU settings for each worker simulation are printed.

Looking at ``libE_stats.txt`` will provide a summary of the runs.

Now try running::

    ./cleanup.sh
    python run_libe_forces.py -n 7

And you will see it runs with two cores and two GPUs are used per
worker. The **forces** example automatically uses the GPUs available to
each worker.

Live viewing GPU usage
----------------------

To see GPU usage, SSH into a compute node you are on in another window and run::

    module load xpu-smi
    watch -n 0.1 xpu-smi dump -d -1 -m 0 -n 1

Using tiles as GPUs
-------------------

To treat each tile as its own GPU, add the ``use_tiles_as_gpus=True`` option
to the ``libE_specs`` block in **run_libe_forces.py**:

.. code-block:: python

    ensemble.libE_specs = LibeSpecs(
        num_resource_sets=nsim_workers,
        sim_dirs_make=True,
        use_tiles_as_gpus=True,
    )

Now you can run again but with twice the workers for running simulations (each
will use one GPU tile)::

    python run_libe_forces.py -n 25

Running generator on the manager
--------------------------------

An alternative is to run the generator on a thread on the manager. The
number of workers can then be set to the number of simulation workers.

Change the ``libE_specs`` in **run_libe_forces.py** as follows:

.. code-block:: python

    nsim_workers = ensemble.nworkers

    # Persistent gen does not need resources
    ensemble.libE_specs = LibeSpecs(
        gen_on_manager=True,

then we can run with 12 (instead of 13) workers::

    python run_libe_forces.py -n 12

Dynamic resource assignment
---------------------------

In the **forces** directory you will also find:

* ``forces_gpu_var_resources`` uses varying processor/GPU counts per simulation.
* ``forces_multi_app`` uses varying processor/GPU counts per simulation and also
  uses two different user executables, one which is CPU-only and one which
  uses GPUs. This allows highly efficient use of nodes for multi-application
  ensembles.

Demonstration
-------------

Note that a video demonstration_ of the *forces_gpu* example on **Frontier**
is also available. The workflow is identical when running on Aurora, with the
exception of different compiler options and numbers of workers (because the
numbers of GPUs on a node differs).

.. _ALCF: https://www.alcf.anl.gov/
.. _Aurora: https://www.alcf.anl.gov/support-center/aurorasunspot/getting-started-aurora
.. _demonstration: https://youtu.be/H2fmbZ6DnVc
