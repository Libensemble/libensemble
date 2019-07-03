==========================
Frequently Asked Questions
==========================

If you have any further questions, feel free to contact us through Support_

.. _Support: https://libensemble.readthedocs.io/en/latest/quickstart.html#support

Parallel Debugging
------------------

**How can I perform parallel debugging on libEnsemble, or debug specific processes?**


Try the following: ``mpiexec -np [num processes] xterm -e 'python [calling script].py'``

This will launch an xterm terminal window specific to each process. Mac users will
need to install xQuartz_.

.. _xQuartz: https://www.xquartz.org/


AssertionError - Idle workers
-----------------------------

**"AssertionError: Should not wait for workers when all workers are idle."**

with ``mpiexec -np 1 python [calling script].py``

This error occurs when the manager is waiting, although no workers are busy.
In the above case, this occurs because an MPI libEnsemble run was initiated with
only one process, resulting in one manager but no workers.

Note: this may also occur with two processes if you are using a persistent generator.
The generator will occupy the one worker, leaving none to run simulation functions.


PETSc and MPI errors
--------------------

**PETSc and MPI errors with "[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=59"**

with ``python [test with PETSc].py --comms local --nworkers 4``

This error occurs on some platforms, including Travis, when using PETSc with libEnsemble
in 'local' (multiprocessing) mode. We believe this is due to PETSc initializing MPI
before libEnsemble forks processes using multiprocessing. The recommended solution
is running libEnsemble in MPI mode. An alternative solution may be using a serial
build of PETSc.

Note: This error does not occur on all platforms and may depend on how multiprocessing
handles an existing MPI communicator in a particular platform.


Fatal error in MPI_Init_thread
------------------------------

**"Fatal error in MPI_Init_thread: Other MPI error, error stack: ... gethostbyname failed"**


This error may be macOS specific. MPI uses TCP to initiate connections,
and needs the local hostname to function. MPI checks /etc/hosts for this information,
and causes the above error if it can't find the correct entry.

Resolve this by appending ``127.0.0.1   [your hostname]`` to /etc/hosts.
Unfortunately, ``127.0.0.1   localhost`` isn't satisfactory for preventing this
error.


macOS - Firewall prompts
------------------------

**macOS - System constantly prompts Firewall Security Permission windows throughout execution**


There are several ways to address this nuisance. One easy (but insecure) solution is
temporarily disabling the Firewall through System Preferences -> Security & Privacy
-> Firewall -> Turn Off Firewall. Alternatively, adding a Firewall "Allow incoming
connections" rule can be tried for the offending Python installations,
but this may not prevent the prompts and only clear them shortly after appearing.
Finally, `Signing your Python installation with a self-signed certificate`_ may
be effective.

.. _`Signing your Python installation with a self-signed certificate`: https://coderwall.com/p/5b_apq/stop-mac-os-x-firewall-from-prompting-with-python-in-virtualenv
