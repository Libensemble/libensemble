.. _datastruct-calc-status:

calc_status
===========

The ``calc_status`` is similar to an exit code, and is either a built-in integer
attribute with a named (enumerated) value and corresponding description, or a
user-specified string. They are determined within user functions to capture the
status of a calculation, and returned to the manager and printed to the ``libE_stats.txt`` file.
Only the status values ``FINISHED_PERSISTENT_SIM_TAG`` and ``FINISHED_PERSISTENT_GEN_TAG``
are currently used by the manager, but others can still provide a useful summary in ``libE_stats.txt``.
The user is responsible for determining the status of a user function instance, since
a given instance could include multiple application runs of mixed rates of success.

The ``calc_status`` is the third optional return value from a user function.
Built-in codes are available in the ``libensemble.message_numbers`` module, but
users are also free to return any custom string.

Example of ``calc_status`` used along with :ref:`Executor<executor_index>` in sim_f:

.. code-block:: python
  :linenos:
  :emphasize-lines: 4,16,19,22,30

    from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED

    task = exctr.submit(calc_type='sim', num_procs=cores, wait_on_start=True)
    calc_status = UNSET_TAG
    poll_interval = 1  # secs
    while(not task.finished):
        if task.runtime > time_limit:
            task.kill()  # Timeout
        else:
            time.sleep(poll_interval)
            task.poll()

    if task.finished:
        if task.state == 'FINISHED':
            print("Task {} completed".format(task.name))
            calc_status = WORKER_DONE
        elif task.state == 'FAILED':
            print("Warning: Task {} failed: Error code {}".format(task.name, task.errcode))
            calc_status = TASK_FAILED
        elif task.state == 'USER_KILLED':
            print("Warning: Task {} has been killed".format(task.name))
            calc_status = WORKER_KILL
        else:
            print("Warning: Task {} in unknown state {}. Error code {}".format(task.name, task.state, task.errcode))

    outspecs = sim_specs['out']
    output = np.zeros(1, dtype=outspecs)
    output['energy'][0] = final_energy

    return output, persis_info, calc_status

Example of defining and returning a custom ``calc_status`` if the built-in values
are insufficient:

.. code-block:: python
  :linenos:

    from libensemble.message_numbers import WORKER_DONE, TASK_FAILED

    task = exctr.submit(calc_type='sim', num_procs=cores, wait_on_start=True)

    task.wait(timeout=60)

    file_output = read_task_output(task)
    if task.errcode == 0:
      if "fail" in file_output:
        calc_status = "Task failed successfully?"
      else:
        calc_status = WORKER_DONE
    else:
      calc_status = TASK_FAILED

    outspecs = sim_specs['out']
    output = np.zeros(1, dtype=outspecs)
    output['energy'][0] = final_energy

    return output, persis_info, calc_status

See forces_simf.py_ for a complete example.
See uniform_or_localopt.py_ for an example of using ``FINISHED_PERSISTENT_GEN_TAG``.

Available values:

..  literalinclude:: ../../libensemble/message_numbers.py
    :start-after: first_calc_status_rst_tag
    :end-before: last_calc_status_rst_tag

The corresponding messages printed to the ``libE_stats.txt`` file:

..  literalinclude:: ../../libensemble/message_numbers.py
    :start-at: calc_status_strings
    :end-before: last_calc_status_string_rst_tag

Example segment of libE_stats.txt:

.. code-block:: console

    Worker     1: Calc     0: gen Time: 0.00 Start: 2019-11-19 18:53:43 End: 2019-11-19 18:53:43 Status: Not set
    Worker     1: Calc     1: sim Time: 4.41 Start: 2019-11-19 18:53:43 End: 2019-11-19 18:53:48 Status: Worker killed
    Worker     2: Calc     0: sim Time: 5.42 Start: 2019-11-19 18:53:43 End: 2019-11-19 18:53:49 Status: Completed
    Worker     1: Calc     2: sim Time: 2.41 Start: 2019-11-19 18:53:48 End: 2019-11-19 18:53:50 Status: Worker killed
    Worker     2: Calc     1: sim Time: 2.41 Start: 2019-11-19 18:53:49 End: 2019-11-19 18:53:51 Status: Worker killed
    Worker     1: Calc     3: sim Time: 4.41 Start: 2019-11-19 18:53:50 End: 2019-11-19 18:53:55 Status: Completed
    Worker     2: Calc     2: sim Time: 4.41 Start: 2019-11-19 18:53:51 End: 2019-11-19 18:53:56 Status: Completed
    Worker     1: Calc     4: sim Time: 4.41 Start: 2019-11-19 18:53:55 End: 2019-11-19 18:53:59 Status: Completed
    Worker     2: Calc     3: sim Time: 4.41 Start: 2019-11-19 18:53:56 End: 2019-11-19 18:54:00 Status: Completed

.. _forces_simf.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/scaling_tests/forces/forces_simf.py
.. _uniform_or_localopt.py: https://github.com/Libensemble/libensemble/blob/develop/libensemble/gen_funcs/uniform_or_localopt.py
