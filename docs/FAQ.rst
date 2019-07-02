==========================
Frequently Asked Questions
==========================

If you have any further questions, feel free to contact us through Support_

.. _Support: https://libensemble.readthedocs.io/en/latest/quickstart.html#support


How can I perform parallel debugging on libEnsemble, or debug specific processes?
---------------------------------------------------------------------------------

Try the following: ``mpiexec -np [num processes] xterm -e 'python [calling script].py'``

This will launch an xterm terminal window specific to each process. Mac users will
need to install xQuartz_

.. _xQuartx: https://www.xquartz.org/


"AssertionError: Should not wait for workers when all workers are idle."
------------------------------------------------------------------------

with ``mpiexec -np 1 python myscript.py``

This error occurs when the manager is waiting, even though no workers are busy.
In the above case, this occurs because an MPI libEnsemble run was initiated with
only one process, resulting in one manager but no workers.

Note: this may also occur with two processes if you are using a persistent generator.
This will tie up the one worker, leaving none to run simulation functions.


PETSc and MPI errors with "[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=59"
---------------------------------------------------------------------------------------

with ``python test_chwirut_pounders.py --comms local --nworkers 4``

This error occurs on some platforms, including Travis, when using PETSc with libEnsemble
in 'local' (multiprocessing) mode. We believe this is due to PETSc initializing MPI
before libEnsemble forks processes using multiprocessing. The recommended solution
is to run libEnsemble in MPI mode. An alternative solution may be to use a serial
build of PETSc.

Note: This error does not occur on all platforms and may depend on how multiprocessing
handles an existing MPI communicator in a particular platform.


"Fatal error in MPI_Init_thread: Other MPI error, error stack: ... gethostbyname failed"
----------------------------------------------------------------------------------------

This error may be a macOS specific issue. MPI uses TCP to initiate connections,
and needs the local hostname to function. MPI checks /etc/hosts for this information,
and causes the above error if it can't find the correct entry.

Resolve this by appending ``127.0.0.1   [your hostname]`` to /etc/hosts.
Unfortunately, ``127.0.0.1   localhost`` isn't satisfactory for preventing this error.


macOS - System constantly prompts Firewall Security Permission windows throughout execution
-------------------------------------------------------------------------------------------

This is a gigantic nuisance, and unfortunately the only known way around this is
temporarily disabling the Firewall through System Preferences -> Security & Privacy
-> Firewall -> Turn Off Firewall. A Firewall "Allow incoming connections" rule can
be added for the offending Python executables and installations, but this doesn't
appear to prevent the prompts; it only clears them shortly after they appear.
