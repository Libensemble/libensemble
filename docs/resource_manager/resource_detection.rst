.. _resource_detection:

Resource Detection
==================

The resource manager can detect system resources, and partition
these to workers. The :doc:`MPI Executor<../executor/mpi_executor>`
accesses the resources available to the current worker when launching tasks.

Node-lists are detected by an environment variable on the following systems:

===========  ===========================
Scheduler       Nodelist Env. variable
===========  ===========================
SLURM           SLURM_NODELIST
COBALT          COBALT_PARTNAME
LSF             LSB_HOSTS/LSB_MCPU_HOSTS
PBS             PBS_NODEFILE
===========  ===========================

These environment variable names can be modified via the :ref:`resource_info<resource_info>`
:class:`libE_specs<libensemble.specs.LibeSpecs>` option.

On other systems, you may have to supply a node list in a file called **node_list**
in your run directory. For example, on the ALCF system Cooley_, the session node list
can be obtained as follows::

            cat $COBALT_NODEFILE > node_list

Resource detection can be disabled by setting
``libE_specs["disable_resource_manager"] = True``, and users can supply run
configuration options on the Executor submit line.

This will usually work sufficiently on
systems that have application-level scheduling and queuing (e.g., ``jsrun``).
However, on many cluster and multi-node systems, if the built-in resource
manager is disabled, then runs without a hostlist or machinefile supplied may be
undesirably scheduled to the same nodes.

System detection for resources can be overridden using the :ref:`resource_info<resource_info>`
:class:`libE_specs<libensemble.specs.LibeSpecs>` option.

.. _Cooley: https://www.alcf.anl.gov/alcf-resources/cooley
