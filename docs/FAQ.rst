==========================
Frequently Asked Questions
==========================

If you have any further questions, feel free to contact us through Support_

.. _Support: https://libensemble.readthedocs.io/en/latest/quickstart.html#support

Common Errors
-------------

**I keep getting "Manager only - must be at least one worker (2 MPI tasks)" when
running with multiprocessing and multiple workers specified.**

If your calling script code was switched from MPI to multiprocessing, make sure that
libE_specs is populated with ``comms: local`` and ``nworkers: [num]``.

**What does "AssertionError: Should not wait for workers when all workers are idle."
mean?**

This error occurs when the manager is waiting although no workers are busy, or
an MPI libEnsemble run was initiated with only one process, resulting in one
manager but no workers.

This may also occur with two processes if you are using a persistent generator.
The generator will occupy the one worker, leaving none to run simulation functions.

**I keep getting: "Not enough processors per worker to honor arguments." when
using the job controller. Can I launch jobs anyway?**

Automatic partitioning of resources can be disabled if you want to oversubscribe
(often if testing on a local machine) by configuring the job controller with
``auto_resources=False``. E.g.::

    jobctrl = MPIJobController(auto_resources=False)

Note that the job_controller ``.launch()`` method has a parameter``hyperthreads``
which will attempt to use all hyperthreads/SMT threads available if set to ``True``

**FileExistsError: [Errno 17] File exists: './sim_worker1'**

This can happen when libEnsemble tries to create sim directories that already exist,
or libEnsemble is launched with ``mpiexec`` when the ``libE_specs['comms']`` option is
set to ``local``.

To create differently named sim directories, you can use the ``sim_dir_suffix``
option in :ref:`sim_specs<datastruct-sim-specs>`.

**PETSc and MPI errors with "[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=59"**

with ``python [test with PETSc].py --comms local --nworkers 4``

This error occurs on some platforms, including Travis CI, when using PETSc with libEnsemble
in ``local`` (multiprocessing) mode. We believe this is due to PETSc initializing MPI
before libEnsemble forks processes using multiprocessing. The recommended solution
is running libEnsemble in MPI mode. An alternative solution may be using a serial
build of PETSc.

.. note::
    This error may depend on how multiprocessing handles an existing MPI
    communicator in a particular platform.

HPC Errors and Questions
------------------------

**Why does libEnsemble hang on certain systems when running with MPI?**

This may occur if matching probes, which mpi4py uses by default, are not supported
by the communications fabric. This has been observed with Intels Truescale (TMI)
fabric at time of writing. This can be solved by switching fabrics or disabling
matching probes before the MPI module is first imported.

Add these two lines BEFORE ``from mpi4py import MPI``::

    import mpi4py
    mpi4py.rc.recv_mprobe = False

Also see https://software.intel.com/en-us/articles/python-mpi4py-on-intel-true-scale-and-omni-path-clusters

**can't open hfi unit: -1 (err=23)**
**[13] MPI startup(): tmi fabric is not available and fallback fabric is not enabled**

This may occur on TMI when libEnsemble Python processes have been launched to a node and these,
in turn, launch jobs on the node; creating too many processes for the available contexts. Note that
while processes can share contexts, the system is confused by the fact that there are two
phases, first libEnsemble processes and then sub-processes to run user jobs. The solution is to
either reduce the number processes running or to specify a fallback fabric through environment
variables::

    unset I_MPI_FABRICS
    export I_MPI_FABRICS_LIST=tmi,tcp
    export I_MPI_FALLBACK=1

Alternatively, libEnsemble can be run in central mode where all workers run on dedicated
nodes, while launching all sub-jobs onto other nodes.

**What does "_pickle.UnpicklingError: invalid load key, '\x00'." indicate?**

This has been observed with the OFA fabric, and usually indicates MPI messages
aren't being received correctly when using mpi4py. The solution
is to either switch fabric or turn off matching probes. See the answer for "Why
does libEnsemble hang on certain systems when running with MPI?"

For more information see: https://bitbucket.org/mpi4py/mpi4py/issues/102/unpicklingerror-on-commrecv-after-iprobe

libEnsemble Help
----------------

**How can I debug specific libEnsemble processes?**

This is most easily addressed when running libEnsemble locally. Try

 ``mpiexec -np [num processes] xterm -e 'python [calling script].py'``

to launch an xterm terminal window specific to each process. Mac users will
need to install xQuartz_.

If running in ``local`` mode try using one of the ``ForkablePdb``
routines in ``libensemble/util/forkpdb.py`` to set breakpoints and debug similarly
to ``pdb``. How well this works varies by system::

    from libensemble.util.forkpdb import ForkablePdb
    ForkablePdb().set_trace()

.. _xQuartz: https://www.xquartz.org/

**Can I use the MPI Job Controller when running libEnsemble with multiprocessing?**

Actually, yes! The job controller type only determines how libEnsemble workers
launch and interact with user applications, and is independent of ``comms`` chosen
for manager-worker communications.

macOS-specific Errors
---------------------

**"Fatal error in MPI_Init_thread: Other MPI error, error stack: ... gethostbyname failed"**

Resolve this by appending ``127.0.0.1   [your hostname]`` to /etc/hosts.
Unfortunately, ``127.0.0.1   localhost`` isn't satisfactory for preventing this
error.

**How do I stop the Firewall Security popups when running with the Job Controller?**

There are several ways to address this nuisance, but all involve trial and error.
An easy (but insecure) solution is temporarily disabling the Firewall through
System Preferences -> Security & Privacy -> Firewall -> Turn Off Firewall. Alternatively,
adding a Firewall "Allow incoming connections" rule can be attempted for the offending
Job Controller executable. We've had limited success running
``sudo codesign --force --deep --sign - /path/to/application.app``
on our Job Controller executables, then confirming the next alerts for the executable
and ``mpiexec.hydra``.

**Frozen PETSc installation following a failed wheel build with** ``pip install petsc petsc4py``

Following a failed wheel build for PETSc, the installation process may freeze when
attempting to configure PETSc with the local Fortran compiler if it doesn't exist.
Run the above command again after disabling Fortran configuring with ``export PETSC_CONFIGURE_OPTIONS='--with-fc=0'``
The wheel build will still fail, but PETSc and petsc4py should still install
successfully via setup.py after some time.
