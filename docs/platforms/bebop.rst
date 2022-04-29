=====
Bebop
=====

Bebop_ is a Cray CS400 cluster with Intel Broadwell and Knights Landing compute
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

    (my_env) user@login:~$ CC=mpiicc MPICC=mpiicc pip install mpi4py --no-binary mpi4py

libEnsemble can then be installed via ``pip`` or ``conda``. To install via ``pip``:

.. code-block:: console

    (my_env) user@login:~$ pip install libensemble

To install via ``conda``:

.. code-block:: console

    (my_env) user@login:~$ conda config --add channels conda-forge
    (my_env) user@login:~$ conda install -c conda-forge libensemble

See :doc:`here<../advanced_installation>` for more information on advanced options
for installing libEnsemble.

Job Submission
--------------

Bebop uses Slurm_ for job submission and management. The two commands you'll
likely use the most to run jobs are ``srun`` and ``sbatch`` for running
interactively and batch, respectively.

libEnsemble node-worker affinity is especially flexible on Bebop. By adjusting
``srun`` runtime options_ users may assign multiple libEnsemble  workers to each
allocated node(oversubscription) or assign multiple nodes per worker.

Interactive Runs
^^^^^^^^^^^^^^^^

You can allocate four Knights Landing nodes for thirty minutes through the following::

    salloc -N 4 -p knl -A [username OR project] -t 00:30:00

With your nodes allocated, queue your job to start with four MPI ranks::

    srun -n 4 python calling.py

``mpirun`` should also work. This line launches libEnsemble with a manager and
**three** workers to one allocated compute node, with three nodes available for
the workers to launch calculations with the Executor or a launch command.
This is an example of running in :doc:`centralized<platforms_index>` mode, and,
if using the :doc:`Executor<../executor/mpi_executor>`, libEnsemble should
be initiated with ``libE_specs['dedicated_mode']=True``

.. note::
    When performing a :doc:`distributed<platforms_index>` MPI libEnsemble run
    and not oversubscribing, specify one more MPI process than the number of
    allocated nodes. The manager and first worker run together on a node.

If you would like to interact directly with the compute nodes via a shell,
the following starts a bash session on a Knights Landing node
for thirty minutes::

    srun --pty -A [username OR project] -p knl -t 00:30:00 /bin/bash

.. note::
    You will need to reactivate your conda virtual environment and reload your
    modules! Configuring this routine to occur automatically is recommended.

Batch Runs
^^^^^^^^^^

Batch scripts specify run settings using ``#SBATCH`` statements. A simple example
for a libEnsemble use case running in :doc:`distributed<platforms_index>` MPI
mode on Broadwell nodes resembles the following:

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH -J myjob
    #SBATCH -N 4
    #SBATCH -p bdwall
    #SBATCH -A myproject
    #SBATCH -o myjob.out
    #SBATCH -e myjob.error
    #SBATCH -t 00:15:00

    # These four lines construct a machinefile for the executor and slurm
    srun hostname | sort -u > node_list
    head -n 1 node_list > machinefile.$SLURM_JOBID
    cat node_list >> machinefile.$SLURM_JOBID
    export SLURM_HOSTFILE=machinefile.$SLURM_JOBID

    srun --ntasks 5 python calling_script.py

With this saved as ``myscript.sh``, allocating, configuring, and running libEnsemble
on Bebop is achieved by running ::

    sbatch myscript.sh

Example submission scripts for running on Bebop in distributed and centralized mode
are also given in the :doc:`examples<example_scripts>`.

Debugging Strategies
--------------------

View the status of your submitted jobs with ``squeue``, and cancel jobs with
``scancel <Job ID>``.

Additional Information
----------------------

See the LCRC Bebop docs here_ for more information about Bebop.

.. _Bebop: https://www.lcrc.anl.gov/systems/resources/bebop/
.. _Anaconda: https://www.anaconda.com/distribution/
.. _conda: https://conda.io/en/latest/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Slurm: https://slurm.schedmd.com/
.. _here: https://www.lcrc.anl.gov/for-users/using-lcrc/running-jobs/running-jobs-on-bebop/
.. _options: https://slurm.schedmd.com/srun.html
