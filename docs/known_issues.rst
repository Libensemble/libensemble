Known Issues
============

The following selection describes known bugs, errors, or other difficulties that
may occur when using libEnsemble.

* OpenMPI does not work with direct MPI job launches in mpi4py comms mode, as
  it does not support nested MPI launches (Either use local mode or Balsam job
  controller).
* Local comms mode (multiprocessing) may fail if MPI is initialized before
  forking processors. This is thought to be responsible for issues combining
  multiprocessing with PETSc on some platforms.
* Remote detection of logical cores via LSB_HOSTS (e.g., Summit) returns number
  of physical cores as SMT info not available.
* TCP mode does not support:
  1) more than one libEnsemble call in a given script or
  2) the auto-resources option to the job controller.
* libEnsemble may hang on systems with matching probes not enabled on the
  native fabric, like on Intel's Truescale (TMI) fabric for instance. See the
  :doc:`FAQ<FAQ>` for more information.
