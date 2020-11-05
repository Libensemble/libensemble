======================================================
Selective Pending Sim Cancellation with Persistent CWP
======================================================

This tutorial demonstrates libEnsemble's capability to selectively cancel pending
simulations based on asynchronous directives from the *Persistent CWP* calibration
generator function. This capability is critical for this calibration use-case since
it isn't useful for the generator to receive pending, extraneous evaluations
from resources that may be more effectively applied towards critical evaluations.

[JLN: BETTER JUSTIFICATION GOES HERE?]

For a somewhat different approach than libEnsemble's :doc:`other tutorials<tutorials>`,
we'll emphasize the settings, components, and data fields within the calling script and CWP
:ref:`persistent generator<persistent-gens>` that make this capability possible,
rather than outlining a step-by-step process for writing this exact use-case.
Nonetheless, we hope that these selections are inspirational for implementing
similar approaches in other user functions.

Generator Side - Reserved Fields and Point Cancellation
-------------------------------------------------------

Like all generator functions, a local :doc:`History array<../history_array>`
is initialized with a given size and datatype::

    H_o = np.zeros(pre_count, dtype=gen_specs['out'])

``gen_specs['out']`` is defined in our calling script, like all specification
dictionaries explored in previous tutorials. For now, know that ``gen_specs['out']``
contains the following elements (alongside many others)::

    [('sim_id', int), ('cancel', bool), ('kill_sent', bool)]

``'sim_id'`` should be familiar from previous tutorials. While the CWP persistent
generator loops, it detects using a library function if any pending points
(combinations of "thetas" and "xs") distributed for simulation ought to be
cancelled (obviated), then calls ``cancel_row()``::

    r_obviate = obviate_pend_thetas(model, theta, data_status)
    if r_obviate[0].shape[0] > 0:
        cancel_row(pre_count, r_obviate, n_x, data_status, comm)

Where ``pre_count`` is a matrix of "thetas" and "xs", ``r_obviate`` is a selection
of rows to cancel, ``n_x`` is the number of ``x`` values, ``data_status`` describes
the calculation status of each point, and ``comm`` is a communicator object used
to send and receive messages from the Manager.

Within ``cancel_row()``, each row in ``r_obviate`` is iterated over, and if a
point's specific ``data_status`` indicates it has not yet been simulated, it's appended
to a list of ``sim_id``'s to be sent to the Manager for cancellation. This communication
is then accomplished through the ``send_mgr_worker_msg`` persistent generator helper
function. Each of these helper functions is described :ref:`here<p_gen_routines>`.
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
        H_o = np.zeros(len(sim_ids_to_cancel), dtype=[('sim_id', int), ('cancel', bool)])
        H_o['sim_id'] = sim_ids_to_cancel
        H_o['cancel'] = True
        send_mgr_worker_msg(comm, H_o)

Note that a new local History array is defined and populated with the requested ``sim_id``'s
to cancel and have their ``'cancel'`` fields set to ``True``. This is then sent
to the manager.

Manager Side - Cancellation Signals and History Updates
-------------------------------------------------------

On the side of the manager, between routines to call the allocation function and
distribute allocated work to each worker, the manager selects points from the History
array that:

    1) Have been marked as ``'given'`` by the allocation function
    2) Have been marked to ``'cancel'`` by the generator
    3) Have *not* been marked as ``'returned'`` by the manager
    4) Have *not* had a kill signal sent by the manager and marked with ``'kill_sent'``

If any points match these characteristics, the workers that are noted as currently
processing these points are sent ``STOP`` tags and a kill signal. Then, ``'kill_sent'``
is marked ``True`` for each of these points in the manager's History array.
