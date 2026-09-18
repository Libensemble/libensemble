.. _executor_index:

**Overview** || :doc:`Base Executor <ex_base>` || :doc:`MPI Executor <ex_mpi>` || :doc:`Flux Executor <ex_flux>` || :doc:`Globus Compute Executor <ex_globus_compute>`

Executors
=========

libEnsemble's Executors can be used within user functions to provide a simple,
portable interface for running and managing user applications.

.. toctree::
    :hidden:

    ex_overview
    ex_base
    ex_mpi
    ex_flux
    ex_globus_compute

The **Executor** provides a portable interface for running applications on any system and
any number of compute resources. The :doc:`MPI Executor<ex_mpi>` launches MPI
applications on local resources; the
:doc:`Globus Compute Executor<ex_globus_compute>` submits Python callables to
remote Globus Compute endpoints.

Please select from the sections above or the sidebar navigation to read more.
