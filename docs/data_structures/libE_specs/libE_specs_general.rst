General
=======

`Introduction <libE_specs.html>`__ \|\| **General** \|\| `Directories <libE_specs_directories.html>`__ \|\| `Profiling <libE_specs_profiling.html>`__ \|\| `TCP <libE_specs_tcp.html>`__ \|\| `History <libE_specs_history.html>`__ \|\| `Resources <libE_specs_resources.html>`__

**comms** [str] = ``"mpi"``:
    Manager/Worker communications mode: ``'mpi'``, ``'local'``, ``'threads'``, or ``'tcp'``.
    If ``nworkers`` is specified, then ``local`` comms will be used unless a
    parallel MPI environment is detected.

**nworkers** [int]:
    Number of worker processes in ``"local"``, ``"threads"``, or ``"tcp"``.

**gen_on_worker** [bool] = False
    Instructs Worker process to run generator instead of Manager.

**mpi_comm** [MPI communicator] = ``MPI.COMM_WORLD``:
    libEnsemble MPI communicator.

**dry_run** [bool] = ``False``:
    Whether libEnsemble should immediately exit after validating all inputs.

**abort_on_exception** [bool] = ``True``:
    In MPI mode, whether to call ``MPI_ABORT`` on an exception.
    If ``False``, an exception will be raised by the manager.

**worker_timeout** [int] = ``1``:
    On libEnsemble shutdown, number of seconds after which workers considered timed out,
    then terminated.

**kill_canceled_sims** [bool] = ``False``:
    Try to kill sims with ``cancel_requested`` set to ``True``.
    If ``False``, the manager avoids this moderate overhead.

**disable_log_files** [bool] = ``False``:
    Disable ``ensemble.log`` and ``libE_stats.txt`` log files.

**gen_workers** [list of ints]:
    List of workers that should run only generators. All other workers will run
    only simulator functions.
