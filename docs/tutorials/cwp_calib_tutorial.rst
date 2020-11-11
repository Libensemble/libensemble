===========================================================
Borehole Calibration with Selective Simulation Cancellation
===========================================================

Introduction - Calibration with libEnsemble and CWP
---------------------------------------------------

This tutorial demonstrates libEnsemble's capability to selectively cancel pending
simulations based on instructions from the *Persistent CWP* calibration
generator function. This capability is critical for this calibration use-case since
it isn't useful for the generator to receive pending, extraneous evaluations
from resources that may be more effectively applied towards critical evaluations.

[JLN: BETTER JUSTIFICATION GOES HERE?]

For a somewhat different approach than libEnsemble's :doc:`other tutorials<tutorials>`,
we'll emphasize the settings, functions, and data fields within the calling script, CWP
:ref:`persistent generator<persistent-gens>`, ``Manager``, and :ref:`sim_f<api_sim_f>`
that make this capability possible, rather than outlining a step-by-step process
for writing this exact use-case.

.. note::
    The generator function featured in this tutorial can be found in ``gen_funcs/persistent_cwp_calib.py``. This version uses simplified standalone routines in place of the in-development CWP library.

Generator - Point Cancellation Requests and Dedicated Fields
------------------------------------------------------------

Given "observed values" at a given set of points ("x"s), the CWP generator seeks to fit
a Gaussian process model to these points using a function parameterized with
"Thetas". The goal is to find the Theta that most closely matches observed values.

After an initial batch of randomly sampled values, the model is used to generate
new Thetas. Each Theta is evaluated via the ``sim_f`` at each of the points, until
some threshold is reached. As mentioned previously, we want the capability to cancel
previously-requested but pending evaluations (of Thetas) to improve efficiency.

[JLN: BETTER DESCRIPTION OF PROBLEM GOES HERE?]

While the CWP persistent generator loops and updates it's model based on returned
points from simulations, it detects using a library function if any pending points
and Thetas distributed for simulation are no longer needed to for the model,
and ought to be cancelled (obviated). The generator then calls ``cancel_row()``::

    r_obviate = obviate_pend_thetas(model, theta, data_status)
    if r_obviate[0].shape[0] > 0:
        cancel_row(pre_count, r_obviate, n_x, data_status, comm)

Where ``pre_count`` is a matrix of "thetas" and "x"s, ``r_obviate`` is a selection
of rows to cancel, ``n_x`` is the number of ``x`` values, ``data_status`` describes
the calculation status of each point, and ``comm`` is a communicator object from
:doc:`libE_info<../data_structures/work_dict>` used to send and receive messages from the Manager.

Within ``cancel_row()``, each row in ``r_obviate`` is iterated over, and if a
point's specific ``data_status`` indicates it has not yet been evaluated by a simulation,
it's appended to a list of ``sim_id``'s to be sent to the Manager for cancellation.
A new, separate local :doc:`History array<../history_output>` is defined with the
selected ``'sim_id'`` s and the ``'cancel_requested'`` field set to ``True``. This array is
then sent to the Manager using the ``send_mgr_worker_msg`` persistent generator
helper function. Each of these helper functions is described :ref:`here<p_gen_routines>`.
The entire ``cancel_row()`` routine within Persistent CWP is listed below::

    def cancel_row(pre_count, r, n_x, data_status, comm):
        # Cancel rest of row
        sim_ids_to_cancel = []
        rows = np.unique(r)
        for r in rows:
            row_offset = r*n_x
            for i in range(n_x):
                sim_id_cancl = pre_count + row_offset + i
                if data_status[r, i] == 0:
                    sim_ids_to_cancel.append(sim_id_cancl)
                    data_status[r, i] = -2

        # Send only these fields to existing H row and it will slot in change.
        H_o = np.zeros(len(sim_ids_to_cancel), dtype=[('sim_id', int), ('cancel_requested', bool)])
        H_o['sim_id'] = sim_ids_to_cancel
        H_o['cancel_requested'] = True
        send_mgr_worker_msg(comm, H_o)

While most Workers, including those running other persistent generators, are only
allocated work when they're in an :ref:`idle or non-active state<../data_structures/worker_array>`,
the CWP generator performs an irregular sending / receiving of data from the Manager
and must be prepared to send or receive data at any moment.
This is necessary since the generator asynchronously updates its model and
cancels pending evaluations. Therefore, the Worker running this generator remains
in a unique *active receive* state, until it becomes non-persistent.

Manager - Cancellation, History Updates, and Allocation
-------------------------------------------------------

On the side of the Manager, between routines to call the allocation function and
distribute allocated work to each Worker, the Manager selects points from the History
array that:

    1) Have been marked as ``'given'`` by the allocation function
    2) Have been marked to ``'cancel_requested'`` by the generator
    3) Have *not* been marked as ``'returned'`` by the Manager
    4) Have *not* been marked with ``'kill_sent'`` by the Manager

If any points match these characteristics, the Workers that are noted as currently
processing these points are sent ``STOP`` tags and a kill signal. Then, ``'kill_sent'``
is marked ``True`` for each of these points in the Manager's History array. During
the subsequent :ref:`start_only_persistent<start_only_persistent_label>` allocation
function calls, any points in the Manager's History array that have ``'cancel_requested'``
as ``True`` are not allocated::

    task_avail = ~H['given'] & ~H['cancel_requested']

This ``alloc_f`` also has the capability to first allocate those points that have
higher ``'priority'`` values in the local History array, effectively prioritizing
simulations with prioritized points from the ``gen_f``.

Simulator - Receiving Kill Signal and Cancelling Tasks
------------------------------------------------------

Within currently running simulation functions, the :doc:`Executor<../executor/overview>`
has been used to launch simulations based on points from the CWP Persistent generator,
and has entered a routine to loop and check for signals from the Manager::

    H_o = np.zeros(H.shape[0], dtype=sim_specs['out'])
    H_o['f'] = borehole_func(H)  # Delay happens within borehole_func

    if check_for_man_kills:
        calc_status = check_for_kill_recv(sim_specs, libE_info)

The contents of ``check_for_kill_recv()`` resemble::

    exctr = Executor.executor
    start_time = time.time()
    while time.time() - start_time < timeout_time:
        time.sleep(poll_interval)
        exctr.manager_poll()
        if exctr.manager_signal == 'kill':
            exctr.kill(task)
            calc_status = MAN_SIGNAL_KILL
            break

    return calc_status

Where the loop periodically sleeps then polls for signals from the Manager using
the :ref:`executor.manager_poll()<manager_poll_label>` function. Notice above that
immediately after ``exctr.manager_signal`` is confirmed as ``'kill'``, the current
task launched by the Executor is killed with the loop breaking and the function
returning with the ``MAN_SIGNAL_KILL`` :doc:`calc_status<../data_structures/calc_status>`.
This status will be logged in ``libE_stats.txt``.

Calling Script - Reading Results
--------------------------------

Within the libEnsemble calling script, once the main :doc:`libE()<../libe_module>`
function call has returned, it's a simple enough process to view the History rows
that were either marked as cancelled and/or had a kill signal sent to their
associated simulation instances during the run::

    H, persis_info, flag = libE(sim_specs, gen_specs,
                                exit_criteria, persis_info,
                                alloc_specs=alloc_specs,
                                libE_specs=libE_specs)

    if is_master:
        print('Cancelled sims', H[H['cancel_requested']])
        print('Killed sims', H[H['kill_sent']])

Please see the ``test_cwp_calib.py`` regression test for an example
routine using the Persistent CWP calibration persistent generator.
The associated simulation function and allocation functions are included in
``sim_funcs/cwpsim.py`` and ``alloc_funcs/start_only_persistent.py`` respectively.
