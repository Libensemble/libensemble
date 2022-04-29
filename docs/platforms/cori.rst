====
Cori
====

Cori_ is a Cray XC40 located at NERSC, featuring both Intel Haswell
and Knights Landing compute nodes. It uses the SLURM scheduler to submit
jobs from login nodes to run on the compute nodes.

Cori does not allow more than one MPI application per compute node.

Configuring Python and Installation
-----------------------------------

Begin by loading the Python 3 Anaconda_ module::

    module load python

Create a conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create a conda_ environment in which to install libEnsemble and
all dependencies. If using ``mpi4py``, it is recommended that you clone
the ``lazy-mpi4py`` environment provided by NERSC::

    conda create --name my_env --clone lazy-mpi4py

If you wish to build ``mpi4py``, it will need to be done using the
specific `Python instructions from NERSC`_.

Installing libEnsemble
----------------------

Having loaded the Anaconda Python module, libEnsemble can be installed
by one of the following ways.

1. Install via **pip** into the environment.

.. code-block:: console

    (my_env) user@cori07:~$ pip install libensemble

2. Install via **conda**:

.. code-block:: console

    (my_env) user@cori07:~$ conda config --add channels conda-forge
    (my_env) user@cori07:~$ conda install -c conda-forge libensemble

It is preferable to create your conda environment under the
``/global/common`` file system, which performs best for imported Python
packages. This can be done by modifying your ``~/.condarc`` file. For
example, add the lines::

    envs_dirs:
      - /path/to/my/conda_envs
    env_prompt: ({name})

The ``env_prompt`` line ensures the whole directory path is not prepended to
your prompt (The ``({name})`` here is literal, do not substitute).

See :doc:`here<../advanced_installation>` for more information on advanced options
for installing libEnsemble.

Job Submission
--------------

Cori uses Slurm_ for job submission and management. The two commands you'll
likely use the most to initiate jobs are ``salloc`` and ``sbatch`` for running
interactively and batch, respectively. libEnsemble runs on the compute nodes
on Cori using either ``multi-processing`` or ``mpi4py``. We recommend reading
the `Python instructions from NERSC`_ for specific guidance on using both
``multiprocessing``(used by local mode in libEnsemble) and ``mpi4py``.

.. note::
    While it is possible to submit jobs from the user ``$HOME`` file system, this
    is likely to perform very poorly, especially for large ensembles. Users
    should preferably submit their calling script from the user
    $SCRATCH (``/global/cscratch1/sd/<YourUserName>``) directory (fastest but
    regularly purged) or the project directory (``/project/projectdirs/<project_name>/``).
    You cannot run and create output under the ``/global/common/`` file system
    as this is read-only from compute nodes, but any imported codes (including
    libEnsemble and gen/sim functions) are best imported from there, especially
    when running at scale.

Interactive Runs
^^^^^^^^^^^^^^^^

You can allocate four Knights Landing nodes for thirty minutes through the following::

    salloc -N 4 -C knl -q interactive -t 00:30:00

Ensure that the Python 3 Anaconda module module is loaded. If you have installed
libEnsemble under the ``common`` file system, ensure ``PYTHONPATH`` is set (as above).

With your nodes allocated, queue your job to start with four MPI ranks::

    srun --ntasks 4 --nodes=1 python calling.py

This line launches libEnsemble with a manager and **three** workers to one
allocated compute node, with three nodes available for the workers to launch
user applications (via the Executor or a direct run command such as ``mpiexec``).

This is an example of running in :doc:`centralized<platforms_index>` mode;
if using the :doc:`Executor<../executor/ex_index>`, libEnsemble should
be initiated with ``libE_specs['dedicated_mode']=True``. libEnsemble must be run in central mode
on Cori because jobs cannot share nodes.

Batch Runs
^^^^^^^^^^

Batch scripts specify run settings using ``#SBATCH`` statements. A simple example
for a libEnsemble use case running in :doc:`centralized<platforms_index>` MPI
mode on KNL nodes resembles the following (add ``PYTHONPATH`` lines if necessary):

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH -J myjob
    #SBATCH -N 5
    #SBATCH -q debug
    #SBATCH -A myproject
    #SBATCH -o myjob.out
    #SBATCH -e myjob.error
    #SBATCH -t 00:15:00
    #SBATCH -C knl

    module load python/3.7-anaconda-2019.07
    export I_MPI_FABRICS=shm:ofi  # Recommend OFI

    # Run libEnsemble (manager and 4 workers) on one node
    # leaving 4 nodes for worker launched applications.
    srun --ntasks 5 --nodes=1 python calling_script.py

With this saved as ``myscript.sh``, allocating, configuring, and running libEnsemble
on Cori is achieved by running ::

    sbatch myscript.sh

If you wish to run in multiprocessing (local) mode instead of using ``mpi4py``
and if your calling script uses the :doc:`parse_args()<../utilities>` function,
then the run line in the above script would be::

    python calling_script.py --comms local --nworkers 4

As a larger example, the following script would launch libEnsemble in MPI mode
with one manager and 128 workers, where each worker will have two nodes for the
user application. libEnsemble could be run on more than one node, but here the
``overcommit`` option to srun is used on one node.

.. code-block:: bash
    :linenos:

    #!/bin/bash
    #SBATCH -J my_bigjob
    #SBATCH -N 257
    #SBATCH -q regular
    #SBATCH -A myproject
    #SBATCH -o myjob.out
    #SBATCH -e myjob.error
    #SBATCH -t 01:00:00
    #SBATCH -C knl

    module load python/3.7-anaconda-2019.07
    export I_MPI_FABRICS=shm:ofi  # Recommend OFI

    # Run libEnsemble (manager and 128 workers) on one node
    # leaving 256 nodes for worker launched applications.
    srun --overcommit --ntasks 129 --nodes=1 python calling_script.py

Example submission scripts are also given in the :doc:`examples<example_scripts>`.

Cori FAQ
--------

**Error in `<PATH>/bin/python': break adjusted to free malloc space: 0x0000010000000000**

This error has been encountered on Cori when running with an incorrect
installation of ``mpi4py``. See instructions above.

Additional Information
----------------------

See the NERSC Cori docs here_ for more information about Cori.

.. _Cori: https://docs.nersc.gov/systems/cori/
.. _Anaconda: https://www.anaconda.com/distribution/
.. _conda: https://conda.io/en/latest/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Slurm: https://slurm.schedmd.com/
.. _here: https://docs.nersc.gov/jobs/
.. _options: https://slurm.schedmd.com/srun.html
.. _Python instructions from NERSC: https://docs.nersc.gov/development/languages/python/parallel-python/
