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

If running in ``'local'`` comms mode try using one of the ``ForkablePdb``
routines in ``libensemble/util/forkpdb.py`` to set breakpoints. How well these
work may depend on the system. Usage::

    from libensemble.util.forkpdb import ForkablePdb
    ForkablePdb().set_trace()


AssertionError - Idle workers
-----------------------------

**"AssertionError: Should not wait for workers when all workers are idle."**

with ``mpiexec -np 1 python [calling script].py``

This error occurs when the manager is waiting, although no workers are busy.
In the above case, this occurs because an MPI libEnsemble run was initiated with
only one process, resulting in one manager but no workers.

Note: this may also occur with two processes if you are using a persistent generator.
The generator will occupy the one worker, leaving none to run simulation functions.


Not enough processors per worker to honour arguments
----------------------------------------------------

**"libensemble.resources.ResourcesException: Not enough processors per worker to honour arguments."**

This is likely when using the job_controller, when there are not enough
cores/nodes available to launch jobs. This can be disabled if you want
to oversubscribe (often if testing on a local machine). Set up the
job_controller with ``auto_resources=False``. E.g.::

    jobctrl = MPIJobController(auto_resources=False)

Also, note that the job_controller launch command has the argument
hyperthreads, which is set to True, will attempt to use all
hyperthreads/SMT threads available.


FileExistsError
---------------

**"FileExistsError: [Errno 17] File exists: './sim_worker1'"**

This can happen when libEnsemble tries to create sim directories that already exist. If
the directory does not already exist, a possible cause is that you are trying
to run using ``mpiexec``, when the ``libE_specs['comms']`` option is set to ``'local'``.
Note that to run with differently named sub-directories you can use the
``'sim_dir_suffix'`` option to :ref:`sim_specs<datastruct-sim-specs>`.


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

**macOS - Constant Firewall Security permission windows throughout Job Controller task**


There are several ways to address this nuisance, but all involve trial and error.
One easy (but insecure) solution is temporarily disabling the Firewall through System Preferences
-> Security & Privacy -> Firewall -> Turn Off Firewall. Alternatively, adding a Firewall "Allow incoming
connections" rule can be tried for the offending Job Controller executable.
Based on a suggestion from here here_, we've had the most success running
``sudo codesign --force --deep --sign - /path/to/application.app`` on our Job Controller executables,
then confirming the next alerts for the executable and ``mpiexec.hydra``.

.. _`here`: https://coderwall.com/p/5b_apq/stop-mac-os-x-firewall-from-prompting-with-python-in-virtualenv


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


macOS - PETSc Installation issues
---------------------------------

**Frozen PETSc installation following a failed wheel build with** ``pip install petsc petsc4py``

Following a failed wheel build for PETSc, the installation process may freeze when
attempting to configure PETSc with the local Fortran compiler if it doesn't exist.
Run the above command again after disabling Fortran configuring with ``export PETSC_CONFIGURE_OPTIONS='--with-fc=0'``
The wheel build will still fail, but PETSc and petsc4py should still install
successfully via setup.py after some time.
