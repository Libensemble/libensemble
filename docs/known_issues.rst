Known Issues
============

The following selection describes known bugs, errors, or other difficulties that
may occur when using libEnsemble.

* As of 10/13/2022, on Perlmutter there was an issue running concurrent applications
  on a node, following a recent system update. This also affects previous versions
  of libEnsemble, and is being investigated.
* When using the Executor: OpenMPI does not work with direct MPI task
  submissions in mpi4py comms mode, since OpenMPI does not support nested MPI
  executions. Use either local mode or the Balsam Executor instead.
* Local comms mode (multiprocessing) may fail if MPI is initialized before
  forking processors. This is thought to be responsible for issues combining
  multiprocessing with PETSc on some platforms.
* Remote detection of logical cores via LSB_HOSTS (e.g., Summit) returns the
  number of physical cores as SMT info not available.
* TCP mode does not support
  (1) more than one libEnsemble call in a given script or
  (2) the auto-resources option to the Executor.
* libEnsemble may hang on systems with matching probes not enabled on the
  native fabric, like on Intel's Truescale (TMI) fabric for instance. See the
  :doc:`FAQ<FAQ>` for more information.
* We currently recommended running in Central mode on Bridges as distributed
  runs are experiencing hangs.
