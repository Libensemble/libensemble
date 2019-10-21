=====
Theta
=====

Theta_ is a 11.69 petaflops system based on the second-generation Intel Xeon Phi
processor, available within ALCF at Argonne National Laboratory.



Before getting started
----------------------

An Argonne ALCF_ account is required to access Theta. Interested users will need
to apply for and be granted an account before continuing. To submit jobs to Theta,
users must charge their jobs to a project (likely specified while requesting an
account.)

Theta consists of numerous types of nodes, but libEnsemble users will mostly interact
with login, Machine Oriented Mini-server (MOM) and compute nodes. MOM nodes execute
user batch scripts to run on the compute nodes.

Configuring Python
------------------



Installing libEnsemble and Dependencies
---------------------------------------


Balsam
^^^^^^


Job Submission
--------------

Theta features one default production queue, ``default``, and two debug queues,
``debug-cache-quad`` and ``debug-flat-quad``.

.. note::
    For the default queue, the minimum number of nodes to allocate is 128!

Theta uses Cobalt for job submission and management. The most important two
commands to run and manage jobs are ``qsub`` and ``aprun``, for submitting batch
scripts from the login nodes and launching jobs to the compute nodes respectively.

Because mpi4py doesn't function on the Theta MOM nodes, libEnsemble is commonly
run in multiprocessing mode with all workers running on the MOM nodes. In these
circumstances, libEnsemble's job controller takes responsibility for ``aprun``
submissions of jobs to compute nodes.



Interactive Runs
^^^^^^^^^^^^^^^^

Users can run interactively with ``qsub`` by specifying the ``-I`` flag, similarly
to the following::

    qsub -A [project] -n 128 -q default -t 120 -I

This will place the user on a MOM node. To launch the job to the allocated compute nodes,
run::

    aprun...

.. note::
    You will need to re-activate your conda virtual environment and reload your
    modules! Configuring this routine to occur automatically is recommended.

Batch Runs
^^^^^^^^^^

Batch scripts specify run-settings using ``#COBALT`` statements. A simple example
for a libEnsemble use-case resembles the following:

.. code-block:: bash

    <code>

With this saved as ``myscript.sh``, allocating, configuring, and running libEnsemble
on Theta becomes::

    qsub --mode script myscript.sh


Running with Balsam
^^^^^^^^^^^^^^^^^^^


Debugging Strategies
--------------------

Each of the two debug queues has sixteen nodes apiece. A user can use up to
eight nodes at a time for a maximum of one hour. Allocate nodes on the debug
queue interactively::

    qsub -A [project] -n 4 -q debug-flat-quad -t 60 -I


Additional Information
----------------------

See the ALCF guides_ on XC40 systems for much more information about Theta.

Become more familiar with Balsam here_.



.. _ALCF: https://www.alcf.anl.gov/
.. _Theta: https://www.alcf.anl.gov/theta
.. _guides: https://www.alcf.anl.gov/user-guides/computational-systems
.. _here: https://balsam.readthedocs.io/en/latest/
