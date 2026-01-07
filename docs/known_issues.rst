Known Issues
============

The following selection describes known bugs, errors, or other difficulties that
may occur when using libEnsemble.

* Platforms using SLURM version 23.02 experience a `pickle error`_ when using
  ``mpi4py`` comms. Disabling matching probes via the environment variable
  ``export MPI4PY_RC_RECV_MPROBE=0`` or adding ``mpi4py.rc.recv_mprobe = False``
  at the top of the calling script should resolve this error. If using the MPI
  executor and multiple workers per node, some users may experience failed
  applications with the message
  ``srun: error: CPU binding outside of job step allocation, allocated`` in
  the application's standard error. This is being investigated. If this happens
  we recommend using ``local`` comms in place of ``mpi4py``.
* When using the Executor: Open-MPI does not work with direct MPI task
  submissions in mpi4py comms mode, since Open-MPI does not support nested MPI
  executions. Use ``local`` mode instead.
* Local comms mode (multiprocessing) may fail if MPI is initialized before
  forking processors. This is thought to be responsible for issues combining
  multiprocessing with PETSc on some platforms.
* Remote detection of logical cores via ``LSB_HOSTS`` (e.g., Summit) returns the
  number of physical cores as SMT info not available.
* TCP mode does not support
  (1) more than one libEnsemble call in a given script or
  (2) the auto-resources option to the Executor.
* libEnsemble may hang on systems with matching probes not enabled on the
  native fabric, like on Intel's Truescale (TMI) fabric for instance. See the
  :doc:`FAQ<FAQ>` for more information.
* We currently recommended running in Central mode on Bridges as distributed
  runs are experiencing hangs.

.. _pickle error: https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#missing-support-for-matched-proberecv
