======================
libEnsemble with Flux
======================

Flux_ is a flexible, hierarchical resource manager and job scheduler used
on a growing number of systems, including LLNL's El Capitan.

libEnsemble can read Flux resource lists and partition these to workers. By
default this is done by :ref:`reading an environment variable<resource_detection>`
(``FLUX_URI``), which is then used to query ``flux resource list`` for the
available nodes.

There are two independent ways to run applications under Flux with
libEnsemble; see the :doc:`Flux Executor <../executor/ex_flux>` page for a
full comparison and usage examples of both:

1. Tell the :doc:`MPIExecutor<../executor/ex_index>` to use ``flux run`` as its
   launcher, either directly::

        from libensemble.executors import MPIExecutor
        exctr = MPIExecutor(custom_info={"mpi_runner": "flux"})

   or via the built-in ``flux`` platform::

        libE_specs["platform"] = "flux"

2. Use the native :doc:`FluxExecutor <../executor/ex_flux>`, which submits
   jobs directly to Flux via its Python API instead of subprocessing a
   launcher. This requires the ``flux-core`` Python bindings and is
   particularly useful in containerized environments::

        from libensemble.executors.flux_executor import FluxExecutor
        exctr = FluxExecutor()

.. _Flux: https://flux-framework.org/
