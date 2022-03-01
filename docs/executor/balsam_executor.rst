Balsam Executors
================

Balsam 2 Executor
-----------------

.. automodule:: new_balsam_executor
  :no-undoc-members:

.. autoclass:: NewBalsamExecutor
  :show-inheritance:
  :members: __init__, register_app, submit_allocation, revoke_allocation, submit

.. autoclass:: NewBalsamTask
  :show-inheritance:
  :member-order: bysource
  :members: poll, wait, kill

Balsam 1 MPI Executor
---------------------

.. automodule:: balsam_executor
  :no-undoc-members:

.. autoclass:: BalsamMPIExecutor
  :show-inheritance:
  :inherited-members:
  :member-order: bysource
  :members: __init__, submit, poll, manager_poll, kill, set_kill_mode

.. autoclass:: BalsamTask
  :show-inheritance:
  :member-order: bysource
  :members: workdir_exists, file_exists_in_workdir, read_file_in_workdir, stdout_exists, read_stdout
  :inherited-members:
