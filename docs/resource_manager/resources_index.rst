.. _resources_index:

Resource Manager
================

libEnsemble comes with built-in resource management. This entails the detection
of available resources (e.g., nodelists, core counts, and GPUs), and the allocation
of resources to workers.

It can be disabled by setting ``libE_specs["disable_resource_manager"] = True``.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Resource Manager:

   overview
   resource_detection
   scheduler_module
   worker_resources
