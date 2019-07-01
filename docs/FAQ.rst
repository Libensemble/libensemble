==========================
Frequently Asked Questions
==========================

If you have any further questions, feel free to contact us through Support_

.. _Support: https://libensemble.readthedocs.io/en/latest/quickstart.html#support

How can I watch my libEnsemble routine work in real-time?
---------------------------------------------------------

The log files ``ensemble.log`` and ``libE_stats.txt`` are updated throughout
execution. The first contains logging output, while the second summarizes each
user-calculation as it is performed.


Can I pass multiple gen_f or sim_f functions to libEnsemble?
------------------------------------------------------------

Not at this time, but libEnsemble provides a :doc:`Job Controller<job_controller/overview>`
module that can be used within gen_f or sim_f functions to launch different types
of jobs.


"AssertionError: Should not wait for workers when all workers are idle."
------------------------------------------------------------------------

with ``mpiexec -np 1 python myscript.py``

This error occurs when the manager is waiting, even though no workers are busy.
In the above case, this occurs because an MPI libEnsemble run was initiated with
only one process, resulting in one manager but no workers.

Note: this may also occur with two processes if you are using a persistent generator.
This will tie up the one worker, leaving none to run simulation functions.


Multiple PETSc and MPI errors with "[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=59"
------------------------------------------------------------------------------------------------

with ``python test_chwirut_pounders.py --comms local --nworkers 4``

This error occurs on some platforms, including Travis, when using PETSc with libEnsemble
in 'local' (multiprocessing) mode. We believe this is due to PETSc initializing MPI
before libEnsemble forks processes using multiprocessing. The recommended solution
is to run libEnsemble in MPI mode. An alternative may be to use a serial build of PETSc.

Note: This error does not occur on all platforms and may depend on how multiprocessing
handles an existing MPI communicator in a particular platform.
