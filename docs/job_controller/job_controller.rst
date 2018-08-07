Job Controller Module
=====================

.. automodule:: controller
  :no-undoc-members:

See  :doc:`example<overview>` for usage.
  
JobController Class
-------------------

The JobController should be constructed after registering applications to a Registry::

    jobctl = JobController(registry = registry)

or if using Balsam::

   jobctr = BalsamJobController(registry = registry)

.. autoclass:: JobController
  :member-order: bysource
  :members: __init__, launch, poll, manager_poll, kill, set_kill_mode

.. autoclass:: BalsamJobController
  :show-inheritance:
  :member-order: bysource  
..  :members: __init__, launch, poll, manager_poll, kill, set_kill_mode


Job Class
---------

Jobs are created and returned though the job_controller launch function. Jobs can be passed as arguments
to the job_controller poll and kill functions. Job information can be queired through the job attributes below and the query funcitons. Note that the job attributes are only updated when they are polled (or though other
job controller functions).

.. autoclass:: Job
  :member-order: bysource
  :members: workdir_exists, file_exists_in_workdir, read_file_in_workdir, stdout_exists, read_stdout
  
.. autoclass:: BalsamJob
  :show-inheritance:
  :member-order: bysource
..  :members: workdir_exists, file_exists_in_workdir, read_file_in_workdir, stdout_exists, read_stdout  
..  :inherited-members:
