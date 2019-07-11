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


libEnsemble hangs when using mpi4py
-----------------------------------

One cause of this could be that the communications fabric does not support matching 
probes (part of the MPI 3.0 standard), which mpi4py uses by default. This has been
observed with Intels Truescale (TMI) fabric at time of writing. This can be solved
either by switch fabric or turning off matching probes before the MPI module is first
imported.

Add these two lines BEFORE 'from mpi4py import MPI'::

    import mpi4py
    mpi4py.rc.recv_mprobe = False

Also see https://software.intel.com/en-us/articles/python-mpi4py-on-intel-true-scale-and-omni-path-clusters


Messages are not received correctly when using mpi4py
------------------------------------------------------

This may manifest itself with the following error:

**"_pickle.UnpicklingError: invalid load key, '\x00'."**

or some similar variation. This has been observed with the OFA fabric. The solution
is to either switch fabric or turn off matching probes.

Add these two lines BEFORE 'from mpi4py import MPI'::

    import mpi4py
    mpi4py.rc.recv_mprobe = False

For more information see: https://bitbucket.org/mpi4py/mpi4py/issues/102/unpicklingerror-on-commrecv-after-iprobe


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


Running out of contexts when running libEnsemble in distributed mode on TMI fabric
----------------------------------------------------------------------------------

The error message may be similar to below:

**"can't open hfi unit: -1 (err=23)"**
**"[13] MPI startup(): tmi fabric is not available and fallback fabric is not enabled"**

This may occur on TMI when libEnsemble Python processes have been launched to a node and these, 
in turn, launch jobs on the node; creating too many processes for the available contexts. Note that
while processes can share contexts, the system is confused by the fact that there are two
phases, first libEnsemble processes and then sub-processes to run user jobs. The solution is to 
either reduce the number processes running or to specify a fallback fabric through environment
variables::

    unset I_MPI_FABRICS
    export I_MPI_FABRICS_LIST=tmi,tcp
    export I_MPI_FALLBACK=1

Another alternative is to run libEnsemble in central mode, in which libEnsemble runs on dedicated
nodes, while launching all sub-jobs to other nodes.
