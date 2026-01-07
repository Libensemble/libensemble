==========================
Frequently Asked Questions
==========================

If you have any additional questions, feel free to contact us through Support_.

Debugging
---------

We recommend using the following options to help debug workflows::

    from libensemble import logger
    logger.set_level("DEBUG")
    libE_specs["safe_mode"] = True

To make it easier to debug a generator try setting the **libE_specs** option ``gen_on_manager``.
To do so, add the following to your calling script::

    libE_specs["gen_on_manager"] = True

With this, ``pdb`` breakpoints can be set as usual in the generator.

For more debugging options see "How can I debug specific libEnsemble processes?" below.

Common Errors
-------------

.. dropdown:: **"Manager only - must be at least one worker (2 MPI tasks)" when running with multiprocessing and multiple workers specified.**

  If your code was recently switched from MPI to multiprocessing,
  make sure that :class:`libE_specs<libensemble.specs.LibeSpecs>` is populated
  with ``"comms": "local"`` and ``"nworkers": [int]``.

.. dropdown:: **"AssertionError: alloc_f did not return any work, although all workers are idle."**

  This error occurs when the manager is waiting although all workers are idle.
  Note that a worker can be in a persistent state but is marked as idle
  when it has returned data to the manager and is ready to receive work.

  Some possible causes of this error are:

  - An MPI libEnsemble run was initiated with only one process, resulting in one
    manager but no workers. Similarly, the error may arise when running with only
    two processes when using a persistent generator. The generator will occupy
    one worker, leaving none to run simulation functions.

  - An error in the allocation function. For example, perhaps the allocation
    waiting for all requested evaluations to be returned (e.g., before starting a
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

.. dropdown:: **libensemble.history (MANAGER_WARNING): Giving entries in H0 back to gen. Marking entries in H0 as 'gen_informed' if 'sim_ended'.**

  This warning is harmless. It's saying that as the provided History array is being "reloaded" into the generator, the copy is being slightly modified.

.. dropdown:: **I keep getting: "Not enough processors per worker to honor arguments." when using the Executor. Can I submit tasks to allocated processors anyway?**

  You may have set `enforce_worker_core_bounds` to True when setting
  up the Executor. Also, the resource manager can be completely disabled
  with::

      libE_specs["disable_resource_manager"] = True

  Note that the Executor ``submit()`` method has a parameter ``hyperthreads``
  which will attempt to use all hyperthreads/SMT threads available if set to ``True``.

.. dropdown:: **FileExistsError: [Errno 17] File exists: "./ensemble"**

  This can happen when libEnsemble tries to create ensemble or simulation directories
  that already exist from previous runs. To avoid this, ensure the ensemble directory
  paths are unique by appending some unique value to ``libE_specs["ensemble_dir_path"]``,
  or automatically instruct runs to operate in unique directories via ``libE_specs["use_workflow_dir"] = True``.

.. dropdown:: **PETSc and MPI errors with "[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=59"**

  with ``python [test with PETSc].py --nworkers 4``

  This error occurs on some platforms when using PETSc with libEnsemble
  in ``local`` (multiprocessing) mode. We believe this is due to PETSc initializing MPI
  before libEnsemble forks processes using multiprocessing. The recommended solution
  is running libEnsemble in MPI mode. An alternative solution may be using a serial
  build of PETSc.

  .. note::
      This error may depend on how multiprocessing handles an existing MPI
      communicator in a particular platform.

.. dropdown:: **"UserWarning: Pydantic serializer warnings:Unexpected extra items present in tuple**

  This warning happens with Pydantic 2.6.0. Upgrading to the latest version
  (2.6.1+) should remove the warning.

HPC Errors and Questions
------------------------

.. dropdown:: **Why does libEnsemble hang on certain systems when running with MPI?**

  Another symptom may be the manager only communicating with Worker 1. This issue
  may occur if matching probes, which mpi4py uses by default, are not supported
  by the communications fabric, like Intel's Truescale (TMI) fabric. This can be
  solved by switching fabrics or disabling matching probes before the MPI module
  is first imported.

  Add these two lines BEFORE ``from mpi4py import MPI``::

      import mpi4py
      mpi4py.rc.recv_mprobe = False

  Also see https://software.intel.com/en-us/articles/python-mpi4py-on-intel-true-scale-and-omni-path-clusters.

.. dropdown:: **can't open hfi unit: -1 (err=23) - [13] MPI startup(): tmi fabric is not available and fallback fabric is not enabled**

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
  and add ``libE_specs["dedicated_mode"] = True`` to your calling script.

.. dropdown:: **What does "_pickle.UnpicklingError: invalid load key, "\x00"." indicate?**

  This has been observed with the OFA fabric when using mpi4py and usually
  indicates MPI messages aren't being received correctly. The solution
  is to either switch fabric or turn off matching probes. See the answer to "Why
  does libEnsemble hang on certain systems when running with MPI?"

  For more information see https://bitbucket.org/mpi4py/mpi4py/issues/102/unpicklingerror-on-commrecv-after-iprobe.

.. dropdown:: **srun: Job \*\*\*\*\*\* step creation temporarily disabled, retrying (Requested nodes are busy)**

  Note that this message has been observed on Perlmutter when none of the problems
  below are present, and is likely caused by interference with system processes
  that run between tasks. In this case, it may cause overhead but does not prevent
  correct functioning.

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

libEnsemble Help
----------------

.. dropdown:: **How can I debug specific libEnsemble processes?**

  This is most easily addressed when running libEnsemble locally. Try

  ``mpiexec -np [num processes] xterm -e "python [calling script].py"``

  to launch an xterm terminal window specific to each process. Mac users will
  need to install xQuartz_.

  If running in ``local`` mode, try using one of the ``ForkablePdb``
  routines in ``libensemble.tools`` to set breakpoints and debug similarly
  to ``pdb``. How well this works varies by system. ::

      from libensemble.tools import ForkablePdb
      ForkablePdb().set_trace()

.. dropdown:: **Can I use the MPI Executor when running libEnsemble with multiprocessing?**

  **Yes**. The Executor type determines only how libEnsemble workers
  execute and interact with user applications and is *independent* of ``comms`` chosen
  for manager/worker communications.

.. dropdown:: **How can I disable libEnsemble's output files?**

  Set ``libE_specs["disable_log_files"]`` to ``True``.

  If libEnsemble aborts on an exception, the History array and ``persis_info``
  dictionaries will be dumped. This can be suppressed by
  setting ``libE_specs["save_H_and_persis_on_abort"]`` to ``False``.

  See :doc:`here<history_output_logging>` for more information about these files.

.. dropdown:: **How can I silence libEnsemble or prevent printed warnings?**

  Some logger messages at or above the ``MANAGER_WARNING`` level are mirrored
  to stderr automatically. To disable this, set the minimum stderr displaying level
  to ``CRITICAL`` via the following::

      from libensemble import logger
      logger.set_stderr_level("CRITICAL")

  This effectively puts libEnsemble in silent mode.

  See the :ref:`Logger Configuration<logger_config>` docs for more information.

macOS and Windows Errors
------------------------

.. _faqwindows:

.. dropdown:: **Can I run libEnsemble on Windows?**

  Although we have run many libEnsemble workflows successfully on Windows using
  both MPI and local comms, Windows is not rigorously supported. We highly
  recommend Unix-like systems. Windows tends to produce more platform-specific
  issues that are difficult to reproduce and troubleshoot.

.. dropdown:: **Windows - How can I run libEnsemble with MPI comms?**

  We have run Windows workflows with MPI comms. However, as most MPI
  distributions have either dropped Windows support (MPICH and Open MPI) or are
  no longer being maintained (``msmpi``), we cannot guarantee success.

  We recommend experimenting with the many Unix-like
  emulators, containers, virtual machines, and other such systems. The
  `Installing PETSc On Microsoft Windows`_ documentation contains valuable
  information.

  Otherwise, install ``msmpi`` and ``mpi4py`` from conda and experiment, or use ``local`` comms.

.. dropdown:: **Windows - "A required privilege is not held by the client"**

  Assuming you were trying to use the ``sim_dir_symlink_files`` or ``gen_dir_symlink_files`` options, this indicates that to
  allow libEnsemble to create symlinks, you need to run your current ``cmd`` shell as administrator.

  **"RuntimeError: An attempt has been made to start a new process... this probably means that you are not using fork...
  " if __name__ == "__main__": freeze_support() ...**

  You need to place your main entry point code underneath an ``if __name__ == "__main__":`` block.

  Explanation: Python chooses one of three methods to start new processes when using multiprocessing
  (``--comms local`` with libEnsemble). These are ``"fork"``, ``"spawn"``, and ``"forkserver"``. ``"fork"``
  is the default on Unix, and in our experience is quicker and more reliable, but ``"spawn"`` is the default
  on Windows and macOS (See the `Python multiprocessing docs`_).

  Prior to libEnsemble v0.9.2, if libEnsemble detected macOS, it would automatically switch the multiprocessing
  method to ``"fork"``. We decided to stop doing this to avoid overriding defaults and compatibility issues with
  some libraries.

  If you'd prefer to use ``"fork"`` or not reformat your code, you can set the
  multiprocessing start method by placing
  the following near the top of your calling script::

    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

.. dropdown:: **"macOS - Fatal error in MPI_Init_thread: Other MPI error, error stack: ... gethostbyname failed"**

  Resolve this by appending ``127.0.0.1   [your hostname]`` to /etc/hosts.
  Unfortunately, ``127.0.0.1   localhost`` isn't satisfactory for preventing this.

.. dropdown:: **macOS - How do I stop the Firewall Security popups when running with the Executor?**

  There are several ways to address this nuisance, but all involve trial and error.
  An easy (but insecure) solution is temporarily disabling the firewall through
  System Preferences -> Security & Privacy -> Firewall -> Turn Off Firewall.
  Alternatively, adding a firewall "Allow incoming connections" rule can be
  attempted for the offending executable. We've had limited success running
  ``sudo codesign --force --deep --sign - /path/to/application.app``
  on our executables, then confirming the next alerts for the executable
  and ``mpiexec.hydra``.

.. _Installing PETSc On Microsoft Windows: https://petsc.org/release/install/windows/#recommended-installation-methods
.. _option to srun: https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/
.. _Python multiprocessing docs: https://docs.python.org/3/library/multiprocessing.html
.. _SDF: https://sdf.slac.stanford.edu/public/doc/#/?id=what-is-the-sdf
.. _Support: https://libensemble.readthedocs.io/en/main/introduction.html#resources
.. _xQuartz: https://www.xquartz.org/
