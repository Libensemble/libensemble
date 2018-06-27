Job Controller Module
---------------------
.. automodule:: controller
  :no-undoc-members:
    
.. autoclass:: JobController
  :members: __init__, launch, poll, manager_poll, kill, set_kill_mode, get_job

.. autoclass:: BalsamJobController
  :members: __init__, launch, poll, manager_poll, kill, set_kill_mode, get_job
  
.. autoclass:: Job
  :members: __init__, workdir_exists, file_exists_in_workdir, read_file_in_workdir, stdout_exists, read_stdout
  
.. autoclass:: BalsamJob
  :members: __init__, workdir_exists, file_exists_in_workdir, read_file_in_workdir, stdout_exists, read_stdout
