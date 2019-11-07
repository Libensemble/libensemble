=====
Bebop
=====

Bebop_ is the newest addition to the computational power of LCRC at Argonne
National Laboratory, featuring both Intel Broadwell and Knights Landing nodes.

Before Getting Started
----------------------

An Argonne LCRC_ account is required to access Bebop. Interested users will need
to apply for and be granted an account before continuing. To submit jobs to Bebop,
users can charge hours to a project or their personal allocation (default).

Bebop consists primarily of login and compute nodes. Users start on the login
nodes, and schedule work for execution on the compute nodes.

Configuring Python
------------------

Begin by loading the Python 3 Anaconda_ module::

    module load anaconda3/2018.12

Create a Conda_ virtual environment in which to install libEnsemble and all
dependencies::

    conda config --add channels intel
    conda create --name my_env intelpython3_core python=3
    source activate my_env

Installing libEnsemble and Dependencies
---------------------------------------

You should have an indication that the virtual environment is activated.
Install mpi4py_ and libEnsemble in this environment, making sure to reference
the pre-installed Intel MPI Compiler. Your prompt should be similar to the
following block:

.. code-block:: console

    (my_env) user@beboplogin4:~$ CC=mpiicc MPICC=mpiicc pip install mpi4py --no-binary mpi4py
    (my_env) user@beboplogin4:~$ pip install libensemble

Job Submission
--------------

Bebop uses Slurm_ for job submission and management. The two commands you'll likely use the most
to run jobs are ``srun`` and ``sbatch`` for running interactively and batch, respectively.

Interactive Runs
^^^^^^^^^^^^^^^^

You can allocate four Knights Landing nodes for thirty minutes through the following::

    salloc -N 4 -p knl -A [username OR project] -t 00:30:00

With your nodes allocated, queue your job to start with five MPI ranks::

    srun -n 5 python calling.py

``mpirun`` should also work. This launches a worker on every node, on which every
worker can perform stand-alone calculations or launch jobs through the job controller.

.. note::
    When performing an MPI libEnsemble run and not oversubscribing, specify one
    more MPI process than the number of allocated nodes. The Manager and first
    worker run together on a node.

Batch Runs
^^^^^^^^^^

Batch scripts specify run-settings using ``#SBATCH`` statements. A simple example
for a libEnsemble use-case running on Broadwell nodes resembles the following:

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

    # These four lines construct a machinefile for the job controller and slurm
    srun hostname | sort -u > node_list
    head -n 1 node_list > machinefile.$SLURM_JOBID
    cat node_list >> machinefile.$SLURM_JOBID
    export SLURM_HOSTFILE=machinefile.$SLURM_JOBID

    srun --ntasks 5 python3 calling_script.py

With this saved as ``myscript.sh``, allocating, configuring, and running libEnsemble
on Bebop becomes::

    sbatch myscript.sh

Debugging Strategies
--------------------

View the status of your submitted jobs with ``squeue`` and cancel jobs with ``scancel [jobID]``.

It's not recommended to debug compute-intensive tasks on the login nodes. Instead, start
a bash session on a Knights Landing node for thirty minutes with::

    srun --pty -A [username OR project] -p knl -t 00:30:00 /bin/bash

.. note::
    You will need to re-activate your conda virtual environment and reload your
    modules! Configuring this routine to occur automatically is recommended.

Additional Information
----------------------

See the LCRC Bebop docs here_ for more information about Bebop.

.. _Bebop: https://www.lcrc.anl.gov/systems/resources/bebop/
.. _LCRC: https://www.lcrc.anl.gov
.. _Anaconda: https://www.anaconda.com/distribution/
.. _Conda: https://conda.io/en/latest/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Slurm: https://slurm.schedmd.com/
.. _here: https://www.lcrc.anl.gov/for-users/using-lcrc/running-jobs/running-jobs-on-bebop/
