MPI Executor
============

.. automodule:: mpi_executor
  :no-undoc-members:

.. autoclass:: MPIExecutor
  :show-inheritance:
  :inherited-members:

  .. automethod:: __init__

..  :member-order: bysource
..  :members: __init__, register_calc, submit, manager_poll

Class-specific Attributes
-------------------------

Class-specific attributes can be set directly to alter the behavior of the MPI
Executor. However, they should be used with caution, because they may not
be implemented in other executors.

:max_submit_attempts: (int) Maximum number of launch attempts for a given
                      task. *Default: 5*.
:fail_time: (int or float) *Only if wait_on_run is set.* Maximum run time to failure in
            seconds that results in relaunch. *Default: 2*.
:retry_delay_incr: (int or float) Delay increment between launch attempts in seconds.
            *Default: 5*. (E.g. First retry after 5 seconds, then 10 seconds, then 15, etc...)

Example. To increase resilience against submission failures::

    taskctrl = MPIExecutor()
    taskctrl.max_launch_attempts = 8
    taskctrl.fail_time = 5
    taskctrl.retry_delay_incr = 10

.. _customizer:

Overriding Auto-detection
-------------------------

libEnsemble detects node-lists, MPI runners, and the number of cores on the node through various
means. When using the MPI Executor it is possible to override the detected information using the
``custom_info`` argument. This takes a dictionary of values.

The allowable fields are::

    'mpi_runner' [string]:
        Select runner: 'mpich', 'openmpi', 'aprun', 'srun', 'jsrun', 'custom'
        All except 'custom' relate to runner classes in libEnsemble.
        Custom allows user to define their own run-lines but without parsing
        arguments or making use of auto-resources.
    'runner_name' [string]:
        Runner name: Replaces run command if present. All runners have a default
        except for 'custom'.
    'cores_on_node' [tuple (int,int)]:
        Tuple (physical cores, logical cores) on nodes.
    'subgroup_launch' [Boolean]:
        Whether MPI runs should be initiatied in a new process group. This needs
        to be correct for kills to work correctly. Use the standalone test at
        libensemble/tests/standalone_tests/kill_test to determine correct value
        for a system.
    'node_file' [string]:
        Name of file containing a node-list. Default is 'node_list'.

For example::

    customizer = {'mpi_runner': 'mpich',
                  'runner_name': 'wrapper -x mpich',
                  'cores_on_node': (16, 64),
                  'node_file': 'libe_nodes'}

    from libensemble.executors.mpi_executor import MPIExecutor
    exctr = MPIExecutor(custom_info=customizer)
