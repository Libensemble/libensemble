===========================================================
Borehole Calibration with Selective Simulation Cancellation
===========================================================

Introduction - Calibration with libEnsemble and CWP
---------------------------------------------------

This tutorial demonstrates libEnsemble's capability to selectively cancel pending
simulations based on instructions from the *Persistent CWP* calibration
generator function. This capability is critical for this calibration use-case since
it isn't useful for the generator to receive extraneous evaluations
from resources that may be more effectively applied towards critical evaluations.

[JLN: BETTER JUSTIFICATION GOES HERE?]

For a somewhat different approach than libEnsemble's :doc:`other tutorials<tutorials>`,
we'll emphasize the settings, functions, and data fields within the calling script, CWP
:ref:`persistent generator<persistent-gens>`, Manager, and :ref:`sim_f<api_sim_f>`
that make this capability possible, rather than outlining a step-by-step process
for writing this exact use-case.

.. note::
    The generator function featured in this tutorial can be found in ``gen_funcs/persistent_cwp_calib.py``. This version uses simplified standalone routines in place of the in-development CWP library.

Generator - Point Cancellation Requests and Dedicated Fields
------------------------------------------------------------

Given "observed values" at a given set of points called "x"s, the CWP generator seeks to fit
a Gaussian process model to these points using a function parameterized with
"Thetas". The goal is to find the Theta that most closely matches observed values.

After an initial batch of randomly sampled values, the model generates
new Thetas. Each Theta is evaluated via the ``sim_f`` at each of the points, until
some error threshold is reached.

[JLN: BETTER DESCRIPTION OF PROBLEM GOES HERE?]

While the generator loops and updates the model based on returned
points from simulations, it detects using a library function if any pending points
and Thetas distributed for simulation are no longer needed for the model
and ought to be cancelled (obviated). The generator then calls ``cancel_row()``::

    r_obviate = obviate_pend_thetas(model, theta, data_status)
    if r_obviate[0].shape[0] > 0:
        cancel_row(pre_count, r_obviate, n_x, data_status, comm)

``pre_count`` is a matrix of Thetas and "x"s, ``r_obviate`` is a selection
of rows to cancel, ``n_x`` is the number of ``x`` values, ``data_status`` describes
the calculation status of each point, and ``comm`` is a communicator object from
:doc:`libE_info<../data_structures/work_dict>` used to send and receive messages from the Manager.

Within ``cancel_row()``, each row in ``r_obviate`` is iterated over, and if a
point's ``data_status`` indicates it has not yet been evaluated by a simulation,
it's ``sim_id`` is appended to a list to be sent to the Manager for cancellation.
A new, separate local :doc:`History array<../history_output>` is defined with the
selected ``'sim_id'`` s and the ``'cancel_requested'`` field set to ``True``. This array is
then sent to the Manager using the ``send_mgr_worker_msg`` persistent generator
helper function. Each of these helper functions is described :ref:`here<p_gen_routines>`.
The entire ``cancel_row()`` routine is listed below::

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

Most Workers, including those running other persistent generators, are only
allocated work when they're in an :doc:`idle or non-active state<../data_structures/worker_array>`.
However, since this generator must asynchronously update its model and
cancel pending evaluations, the Worker running this generator remains
in a unique *active receive* state, until it becomes non-persistent. This means
both the Manager and persistent Worker must be prepared for irregular sending /
receiving of data.

Manager - Cancellation, History Updates, and Allocation
-------------------------------------------------------

Between routines to call the allocation function and distribute allocated work
to each Worker, the Manager selects points from the History array that are:

    1) Marked as ``'given'`` by the allocation function
    2) Marked with ``'cancel_requested'`` by the generator
    3) *Not* been marked as ``'returned'`` by the Manager
    4) *Not* been marked with ``'kill_sent'`` by the Manager

If any points match these characteristics, the Workers that are processing these
points are sent ``STOP`` tags and a kill signal. ``'kill_sent'``
is set to ``True`` for each of these points in the Manager's History array. During
the subsequent :ref:`start_only_persistent<start_only_persistent_label>` allocation
function calls, any points in the Manager's History array that have ``'cancel_requested'``
as ``True`` are not allocated::

    task_avail = ~H['given'] & ~H['cancel_requested']

This ``alloc_f`` also can prioritize allocating points that have
higher ``'priority'`` values from the ``gen_f`` values in the local History array::

    # Loop through available simulation workers
    for i in avail_worker_ids(W, persistent=False):

        if np.any(task_avail):
            if 'priority' in H.dtype.fields:
                priorities = H['priority'][task_avail]
                if gen_specs['user'].get('give_all_with_same_priority'):
                    indexes = (priorities == np.max(priorities))
                else:
                    indexes = np.argmax(priorities)
            else:
                indexes = 0

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

The loop periodically sleeps, then polls for signals from the Manager using
the :ref:`executor.manager_poll()<manager_poll_label>` function. Notice that
immediately after ``exctr.manager_signal`` is confirmed as ``'kill'``, the current
task launched by the Executor is killed and the function returns with the
``MAN_SIGNAL_KILL`` :doc:`calc_status<../data_structures/calc_status>`.
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

Here's an example graph showing the relationship between scheduled, cancelled (obviated),
failed, and completed simulations requested by the ``gen_f``. Notice that for each
batch of scheduled simulations, most either complete or fail but the rest are
successfully obviated:

.. image:: ../images/numparam.png
  :alt: cwp_sample_graph

Please see the ``test_cwp_calib.py`` regression test for an example
routine using the Persistent CWP calibration generator.
The associated simulation function and allocation function are included in
``sim_funcs/cwpsim.py`` and ``alloc_funcs/start_only_persistent.py`` respectively.
