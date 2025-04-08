=====
Bebop
=====

Bebop_ is a Cray CS400 cluster with Intel Broadwell compute
nodes available in the Laboratory Computing Resources
Center (LCRC) at Argonne National
Laboratory.

Configuring Python
------------------

Begin by loading the Python 3 Anaconda_ module::

    module load anaconda3

Create a conda_ virtual environment in which to install libEnsemble and all
dependencies::

    conda config --add channels intel
    conda create --name my_env intelpython3_core python=3
    source activate my_env

Installing libEnsemble and Dependencies
---------------------------------------

You should have an indication that the virtual environment is activated.
Start by installing mpi4py_ in this environment, making sure to reference
the preinstalled Intel MPI compiler. Your prompt should be similar to the
following block:

.. code-block:: console

    CC=mpiicc MPICC=mpiicc pip install mpi4py --no-binary mpi4py

libEnsemble can then be installed via ``pip`` or ``conda``. To install via ``pip``:

.. code-block:: console

    pip install libensemble

To install via ``conda``:

.. code-block:: console

    conda config --add channels conda-forge
    conda install -c conda-forge libensemble

See :doc:`here<../advanced_installation>` for more information on advanced options
for installing libEnsemble.

Job Submission
--------------

Bebop uses PBS for job submission and management.

Interactive Runs
^^^^^^^^^^^^^^^^

You can allocate four Broadwell nodes for thirty minutes through the following::

    qsub -I -A <project_id> -l select=4:mpiprocs=4 -l walltime=30:00

Once in the interactive session, you may need to reload your modules::

    cd $PBS_O_WORKDIR
    module load anaconda3 gcc openmpi aocl
    conda activate bebop_libe_env

Now run your script with four workers (one for generator and three for simulations)::

    python my_libe_script.py --nworkers 4

``mpirun`` should also work. This line launches libEnsemble with a manager and
**three** workers to one allocated compute node, with three nodes available for
the workers to launch calculations with the Executor or a launch command.
This is an example of running in :doc:`centralized<platforms_index>` mode, and,
if using the :doc:`Executor<../executor/mpi_executor>`, libEnsemble should
be initiated with ``libE_specs["dedicated_mode"]=True``

.. note::
    When performing a :doc:`distributed<platforms_index>` MPI libEnsemble run
    and not oversubscribing, specify one more MPI process than the number of
    allocated nodes. The manager and first worker run together on a node.

.. note::
    You will need to reactivate your conda virtual environment and reload your
    modules! Configuring this routine to occur automatically is recommended.

Additional Information
----------------------

See the LCRC Bebop docs here_ for more information about Bebop.

.. _Anaconda: https://www.anaconda.com/
.. _Bebop: https://www.lcrc.anl.gov/systems/bebop
.. _conda: https://conda.io/en/latest/
.. _here: https://docs.lcrc.anl.gov/bebop/running-jobs-bebop/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
