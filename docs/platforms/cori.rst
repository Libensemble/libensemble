====
Cori
====

Cori_ is a Cray XC40 located at NERSC, featuring both Intel Haswell
and Knights Landing compute nodes. It uses the SLURM schedular to submit
jobs from login nodes to run on the compute nodes.

Cori does not allow more than one MPI application per compute node.

Configuring Python and installation
-----------------------------------

Begin by loading the Python 3 Anaconda_ module::

    module load python/3.7-anaconda-2019.07

In many cases this may provide all the dependent packages you need (including
mpi4py). Then libEnsemble can be installed locally::

    export PYTHONNOUSERSITE=0
    pip install libensemble --user

Alternatively, you can create your own Conda_ environment in which to install
libEnsemble and all dependencies. If using ``mpi4py``, installation will need
to be done using the `specific instructions from NERSC`_. libEnsemble can then
be pip installed into the environment.

.. code-block:: console

    (my_env) user@cori07:~$ pip install libensemble

If highly parallel runs experience long start-up delays consider the NERSC
documentation on `scaling Python`_.

Job Submission
--------------

Cori uses Slurm_ for job submission and management. The two commands you'll
likely use the most to run jobs are ``srun`` and ``sbatch`` for running
interactively and batch, respectively. libEnsemble runs on the compute nodes
on Cori using either ``multi-processing`` or ``mpi4py``.

Interactive Runs
^^^^^^^^^^^^^^^^

You can allocate four Knights Landing nodes for thirty minutes through the following::

    salloc -N 4 -C knl -q interactive -t 00:30:00

With your nodes allocated, queue your job to start with four MPI ranks::

    srun --ntasks 4 --nodes=1 python calling.py

This line launches libEnsemble with a manager and **three** workers to one
allocated compute node, with three nodes available for the workers to launch
user applications with the job-controller or a job-launch command.

This is an example of running in :doc:`centralized<platforms_index>` mode and,
if using the :doc:`job_controller<../job_controller/mpi_controller>`, it should
be intiated with ``central_mode=True``. libEnsemble must be run in central mode
on Cori as jobs cannot share nodes.

Batch Runs
^^^^^^^^^^

Batch scripts specify run-settings using ``#SBATCH`` statements. A simple example
for a libEnsemble use-case running in :doc:`centralized<platforms_index>` MPI
mode on KNL nodes resembles the following:

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

    # Run libEnsemble (manager and 4 workers) on one node
    # leaving 4 nodes for worker launched applications.
    srun --ntasks 5 --nodes=1 python calling_script.py

With this saved as ``myscript.sh``, allocating, configuring, and running libEnsemble
on Cori becomes::

    sbatch myscript.sh

If you wish to run in multi-processing (local) mode instead of using mpi4py,
and your calling script uses the :doc:`parse_args()<../utilities>` function,
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

    # Run libEnsemble (manager and 128 workers) on one node
    # leaving 256 nodes for worker launched applications.
    srun --overcommit --ntasks 129 --nodes=1 python calling_script.py

Example submission scripts are also given in the examples_ directory.

Additional Information
----------------------

See the NERSC Cori docs here_ for more information about Cori.

.. _Cori: https://docs.nersc.gov/systems/cori/
.. _Anaconda: https://www.anaconda.com/distribution/
.. _Conda: https://conda.io/en/latest/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Slurm: https://slurm.schedmd.com/
.. _here: https://docs.nersc.gov/jobs/
.. _options: https://slurm.schedmd.com/srun.html
.. _examples: https://github.com/Libensemble/libensemble/tree/develop/examples/job_submission_scripts
.. _specific instructions from NERSC: https://docs.nersc.gov/programming/high-level-environments/python/mpi4py/
.. _scaling Python: https://docs.nersc.gov/programming/high-level-environments/python/scaling-up