==========================
Frequently Asked Questions
==========================

If you have any additional questions, feel free to contact us through Support_.

.. _Support: https://libensemble.readthedocs.io/en/latest/quickstart.html#support

Common Errors
-------------

**"Manager only - must be at least one worker (2 MPI tasks)" when
running with multiprocessing and multiple workers specified.**

If your calling script code was recently switched from MPI to multiprocessing,
make sure that ``libE_specs`` is populated with ``comms: local`` and ``nworkers: [num]``.

**"AssertionError: alloc_f did not return any work, although all workers are idle."**

This error occurs when the manager is waiting although all workers are idle.
Note that a worker can be in a persistent state but is marked as idle
when it has returned data to the manager and is ready to receive work.

Some possible causes of this error are:

- An MPI libEnsemble run was initiated with only one process, resulting in one
  manager but no workers. Similarly, the error may arise when running with only
  two processes when using a persistent generator. The generator will occupy
  one worker, leaving none to run simulation functions.

- An error in the allocation function. For example, perhaps the allocation
  waiting for all requested evaluations to be returned (e.g, before starting a
  new generator), but this condition
  is not returning True even though all scheduled evaluations have returned. This
  can be due to incorrect implementation (e.g., it has not considered points that
  are cancelled or paused or in some other state that prevents the allocation
  function from sending them out to workers).

- A persistent worker (usually a generator) has sent a message back to the manager
  but is still performing work and may return further points. In this case, consider
  starting the generator in :ref:`active_recv<gen_active_recv>` mode. This can be
  specified in the allocation function and will cause the worker to maintain its
  active status.

- A persistent worker has requested resources that prevents any simulations from
  taking place. By default, persistent workers hold onto resources even when not
  active. This may require the worker to return from persistent mode.

- When returning points to a persistent generator (often the top code block in
  allocation functions). For example, ``support.avail_worker_ids(persistent=EVAL_GEN_TAG)``
  Make sure that the ``EVAL_GEN_TAG`` is specified and not just ``persistent=True``.

**I keep getting: "Not enough processors per worker to honor arguments." when
using the Executor. Can I submit tasks to allocated processors anyway?**

You may have set `enforce_worker_core_bounds` to True when setting
up the Executor. Also, the resource manager can be completely disabled
with::

    libE_specs['disable_resource_manager'] = True

Note that the Executor ``submit()`` method has a parameter ``hyperthreads``
which will attempt to use all hyperthreads/SMT threads available if set to ``True``.

**FileExistsError: [Errno 17] File exists: './ensemble'**

This can happen when libEnsemble tries to create ensemble or simulation directories
that already exist from previous runs. To avoid this, ensure the ensemble directory
paths are unique by appending some unique value to ``libE_specs['ensemble_dir_path']``

**PETSc and MPI errors with "[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=59"**

with ``python [test with PETSc].py --comms local --nworkers 4``

This error occurs on some platforms when using PETSc with libEnsemble
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

Another symptom may be the manager only communicating with Worker 1. This issue
may occur if matching probes, which mpi4py uses by default, are not supported
by the communications fabric, like Intel's Truescale (TMI) fabric. This can be
solved by switching fabrics or disabling matching probes before the MPI module
is first imported.

Add these two lines BEFORE ``from mpi4py import MPI``::

    import mpi4py
    mpi4py.rc.recv_mprobe = False

Also see https://software.intel.com/en-us/articles/python-mpi4py-on-intel-true-scale-and-omni-path-clusters.

**can't open hfi unit: -1 (err=23)**
**[13] MPI startup(): tmi fabric is not available and fallback fabric is not enabled**

This may occur on TMI when libEnsemble Python processes have been launched to a
node and these, in turn, execute tasks on the node; creating too many processes
for the available contexts. Note that while processes can share contexts, the
system is confused by the fact that there are two phases: first libEnsemble
processes and then subprocesses to run user tasks. The solution is to either
reduce the number of processes running or to specify a fallback fabric through
environment variables::

    unset I_MPI_FABRICS
    export I_MPI_FABRICS_LIST=tmi,tcp
    export I_MPI_FALLBACK=1

Alternatively, libEnsemble can be run in central mode where all workers run on dedicated
nodes while launching all tasks onto other nodes. To do this add a node for libEnsemble,
and add ``libE_specs['dedicated_mode'] = True`` to your calling script.

**What does "_pickle.UnpicklingError: invalid load key, '\x00'." indicate?**

This has been observed with the OFA fabric when using mpi4py and usually
indicates MPI messages aren't being received correctly. The solution
is to either switch fabric or turn off matching probes. See the answer for "Why
does libEnsemble hang on certain systems when running with MPI?"

For more information see https://bitbucket.org/mpi4py/mpi4py/issues/102/unpicklingerror-on-commrecv-after-iprobe.

**Error in `<PATH>/bin/python': break adjusted to free malloc space: 0x0000010000000000**

This error has been encountered on Cori when running with an incorrect installation of ``mpi4py``.
Make sure platform specific instructions are followed (e.g.~ :doc:`Cori<platforms/cori>`)

**srun: Job \*\*\*\*\*\* step creation temporarily disabled, retrying (Requested nodes are busy)**

You may also see: ``srun: Job ****** step creation still disabled, retrying (Requested nodes are busy)``

When running on a SLURM system, this implies that you are trying to run on a resource
that is already dedicated to another task. The reason can vary, some reasons are:

- All the contexts are in use. This has occurred when using TMI fabric on clusters.
  See question **can't open hfi unit: -1 (err=23)** for more info.

- All the memory is assigned to the first job-step (srun application), due to a default
  exclusive mode scheduling policy. This has been observed on `Perlmutter`_ and `SDF`_.

  In some cases using these environment variables will stop the issue::

    export SLURM_EXACT=1
    export SLURM_MEM_PER_NODE=0

  Alternatively, this can be resolved by limiting the memory and other
  resources given to each task using the ``--exact`` `option to srun`_ along with other
  relevant options. For example::

      srun --exact -n 4 -c 1 --mem-per-cpu=4G

  would ensure that one CPU and 4 Gigabytes of memory are assigned to each MPI process.
  The amount of memory should be determined by the memory on the node divided by
  the number of CPUs. In the executor, this can be expressed via the ``extra_args`` option.

  If libEnsemble is sharing nodes with submitted tasks (user applications launched by workers),
  then you may need to do this for your launch of libEnsemble also, ensuring there are enough
  resources for both the libEnsemble manager and workers and the launched tasks. If this is
  complicated, we recommended using a :doc:`dedicated node for libEnsemble<platforms/platforms_index>`.

.. _option to srun: https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter
.. _SDF: https://sdf.slac.stanford.edu/public/doc/#/?id=what-is-the-sdf

libEnsemble Help
----------------

**How can I debug specific libEnsemble processes?**

This is most easily addressed when running libEnsemble locally. Try

 ``mpiexec -np [num processes] xterm -e 'python [calling script].py'``

to launch an xterm terminal window specific to each process. Mac users will
need to install xQuartz_.

If running in ``local`` mode, try using one of the ``ForkablePdb``
routines in ``libensemble.tools`` to set breakpoints and debug similarly
to ``pdb``. How well this works varies by system. ::

    from libensemble.tools import ForkablePdb
    ForkablePdb().set_trace()

.. _xQuartz: https://www.xquartz.org/

**Can I use the MPI Executor when running libEnsemble with multiprocessing?**

Yes. The Executor type determines only how libEnsemble workers
execute and interact with user applications and is independent of ``comms`` chosen
for manager/worker communications.

**How can I disable libEnsemble's output files?**

To disable ``libe_stats.txt`` and ``ensemble.log``, which libEnsemble typically
always creates, set ``libE_specs['disable_log_files']`` to ``True``.

If libEnsemble aborts on an exception, the History array and ``persis_info``
dictionaries will be dumped. This can be suppressed by
setting ``libE_specs['save_H_and_persis_on_abort']`` to ``False``.

See :doc:`here<history_output_logging>` for more information about these files.

**How can I silence libEnsemble or prevent printed warnings?**

Some logger messages at or above the ``MANAGER_WARNING`` level are mirrored
to stderr automatically. To disable this, set the minimum stderr displaying level
to ``CRITICAL`` via the following::

    from libensemble import logger
    logger.set_stderr_level('CRITICAL')

This effectively puts libEnsemble in silent mode.

See the :ref:`Logger Configuration<logger_config>` docs for more information.

macOS-Specific Errors
---------------------

**"Fatal error in MPI_Init_thread: Other MPI error, error stack: ... gethostbyname failed"**

Resolve this by appending ``127.0.0.1   [your hostname]`` to /etc/hosts.
Unfortunately, ``127.0.0.1   localhost`` isn't satisfactory for preventing this
error.

**How do I stop the Firewall Security popups when running with the Executor?**

There are several ways to address this nuisance, but all involve trial and error.
An easy (but insecure) solution is temporarily disabling the firewall through
System Preferences -> Security & Privacy -> Firewall -> Turn Off Firewall.
Alternatively, adding a firewall "Allow incoming connections" rule can be
attempted for the offending executable. We've had limited success running
``sudo codesign --force --deep --sign - /path/to/application.app``
on our Executor executables, then confirming the next alerts for the executable
and ``mpiexec.hydra``.

**Frozen PETSc installation following a failed wheel build with** ``pip install petsc petsc4py``

Following a failed wheel build for PETSc, the installation process may freeze when
attempting to configure PETSc with the local Fortran compiler if it doesn't exist.
Run the above command again after disabling Fortran configuring with ``export PETSC_CONFIGURE_OPTIONS='--with-fc=0'``.
The wheel build will still fail, but PETSc and petsc4py should still install
successfully via ``setup.py`` after some time.
