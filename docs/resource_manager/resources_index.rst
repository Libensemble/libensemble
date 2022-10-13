.. _resources_index:

Resource Manager
================

libEnsemble comes with built-in resource management. This entails the detection of available resources (e.g. nodelists and core counts), and the allocation of resources to workers.

Resource management can be disabled by setting
``libE_specs['disable_resource_manager'] = True``. This will prevent libEnsemble
from doing any resource detection or management.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Resource Manager:

   Zero-resource workers (e.g.,~ Persistent gen does not need resources) <zero_resource_workers>
   overview
   resource_detection
   scheduler_module
   Worker Resources Module (query resources for current worker) <worker_resources>
