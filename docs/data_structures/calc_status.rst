.. _datastruct-calc-status:

calc_status
===========

The ``calc_status`` is an integer attribute with named (enumerated) values and
a corresponding description that can be used in :ref:`sim_f<api_sim_f>` or
:ref:`gen_f<api_gen_f>` functions to capture the status of a calcuation. This
is returned to the manager and printed to the ``libE_stats.txt`` file. Only the
status values ``FINISHED_PERSISTENT_SIM_TAG`` and
``FINISHED_PERSISTENT_GEN_TAG`` are currently used by the manager,  but others
can still provide a useful summary in libE_stats.txt. The user determines the
status of the calculation, as it could include multiple application runs. It
can be added as a third return variable in sim_f or gen_f functions.
The calc_status codes are in the ``libensemble.message_numbers`` module.

Example of ``calc_status`` used along with :ref:`job controller<jobcontroller_index>` in sim_f:

.. code-block:: python
  :linenos:
  :emphasize-lines: 4,16,19,22,30

    from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, JOB_FAILED

    job = jobctl.launch(calc_type='sim', num_procs=cores, wait_on_run=True)
    calc_status = UNSET_TAG
    poll_interval = 1  # secs
    while(not job.finished):
        if job.runtime > time_limit:
            job.kill()  # Timeout
        else:
            time.sleep(poll_interval)
            job.poll()

    if job.finished:
        if job.state == 'FINISHED':
            print("Job {} completed".format(job.name))
            calc_status = WORKER_DONE
        elif job.state == 'FAILED':
            print("Warning: Job {} failed: Error code {}".format(job.name, job.errcode))
            calc_status = JOB_FAILED
        elif job.state == 'USER_KILLED':
            print("Warning: Job {} has been killed".format(job.name))
            calc_status = WORKER_KILL
        else:
            print("Warning: Job {} in unknown state {}. Error code {}".format(job.name, job.state, job.errcode))

    outspecs = sim_specs['out']
    output = np.zeros(1, dtype=outspecs)
    output['energy'][0] = final_energy

    return output, persis_info, calc_status

See forces_simf.py_ for a complete example.
See uniform_or_localopt.py_ for an example of using *FINISHED_PERSISTENT_GEN_TAG*

Available values:

..  literalinclude:: ../../libensemble/message_numbers.py
    :start-after: first_calc_status_rst_tag
    :end-before: last_calc_status_rst_tag

The corresponding messages printed to the ``libE_stats.txt`` file are:

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
