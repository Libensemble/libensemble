Generator Functions
===================

As described in the :ref:`API<api_gen_f>`, the ``gen_f`` is called by a
libEnsemble worker via the following::

    out = gen_f(H[gen_specs['in']][sim_ids_from_allocf], persis_info, gen_specs, libE_info)

In practice, most ``gen_f`` function definitions written by users resemble::

    def my_generator(H, persis_info, gen_specs, libE_info):

Where :doc:`H<../data_structures/history_array>` is a selection of the
:doc:`History array<../history_output_logging>`, determined by sim IDs from the
``alloc_f``, :doc:`persis_info<../data_structures/persis_info>` is a dictionary
containing state information, :doc:`gen_specs<../data_structures/gen_specs>` is a
dictionary containing pre-defined parameters for the ``gen_f``, and ``libE_info``
is a dictionary containing libEnsemble-specific entries. See the API above for
more detailed descriptions of the parameters.

.. note::

    If the ``gen_f`` is a persistent generator, then ``gen_specs['in']`` only specifies
    the fields to send when the ``gen_f`` is *first called.* Use ``gen_specs['persis_in']``
    to specify fields to send back to the generator throughout runtime.

Typically users start by extracting their custom parameters initially defined
within ``gen_specs['user']`` in the calling script and defining a *local* History
array based on the datatype in ``gen_specs['out']``, to be returned. For example::

        batch_size = gen_specs['user']['batch_size']
        local_H_out = np.zeros(batch_size, dtype=gen_specs['out'])

This array should be populated by whatever values are generated within
the function. Finally, this array should be returned to libEnsemble
alongside ``persis_info``::

        return local_H_out, persis_info

Between the output array definition and the function returning, any level and complexity
of computation can be performed. Users are encouraged to use the :doc:`executor<../executor/overview>`
to submit applications to parallel resources if necessary, or plug in components from
any other libraries to serve their needs.

.. note::

    State ``gen_f`` information like checkpointing should be
    appended to ``persis_info``.

.. _persistent-gens:

Persistent Generators
---------------------

While non-persistent generators return after completing their calculation, persistent
generators receive work units, perform computations, and communicate results
directly to the manager in a loop. A persistent generator returns either when
explicitly instructed by the manager, or by exiting its main loop based on some
condition. The allocation function can determine what to do once a persistent
generator finishes, such as ending the ensemble.

The calling worker becomes a dedicated :ref:`persistent worker<persis_worker>`.
A ``gen_f`` is initiated as persistent by the ``alloc_f``.

Many users prefer persistent generators since they do not need to be
re-initialized every time their past work is completed and evaluated by a
simulation, and can evaluate returned simulation results over the course of
an entire libEnsemble routine as a single function instance. The :doc:`APOSMM<../examples/aposmm>`
optimization generator function included with libEnsemble is persistent so it can
maintain multiple local optimization subprocesses based on results from complete simulations.

Functions for a persistent generator to communicate directly with the manager
are available in the :ref:`libensemble.tools.persistent_support<p_gen_routines>` class.
Additional necessary resources are the status tags ``STOP_TAG``, ``PERSIS_STOP``, ``EVAL_GEN_TAG``, and
``FINISHED_PERSISTENT_GEN_TAG`` from ``libensemble.message_numbers``. Return
values from the ``persistent_support`` functions are compared to these tags to determine when
the generator should break its loop and return.

A ``PersistentSupport`` class instance should resemble::

    my_support = PersistentSupport(libE_info, EVAL_GEN_TAG)

Implementing functions from the above class is relatively simple:

.. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
.. autofunction:: send

This function call typically resembles::

    my_support.send(local_H_out[selected_IDs])

Note that this function has no return.

.. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
.. autofunction:: recv

This function call typically resembles::

    tag, Work, calc_in = my_support.recv()

    if tag in [STOP_TAG, PERSIS_STOP]:
        cleanup()
        break

The logic following the function call is typically used to break the persistent
generator's main loop and return.

.. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
.. autofunction:: send_recv

This function performs both of the previous functions in a single statement. Its
usage typically resembles::

    tag, Work, calc_in = my_support.send_recv(local_H_out[selected_IDs])
    if tag in [STOP_TAG, PERSIS_STOP]:
        cleanup()
        break

Once the persistent generator's loop has been broken because of
the tag from the manager, it should return with an additional tag::

    return local_H_out, persis_info, FINISHED_PERSISTENT_GEN_TAG

See :doc:`calc_status<../data_structures/calc_status>` for more information about
the message tags.

.. _gen_active_recv:

Active receive mode
-------------------

By default, a persistent worker (generator in this case) models the manager/worker
communications of a regular worker (i.e., the generator is expected to alternately
receive and send data in a *ping pong* fashion). To have an irregular communication
pattern, a worker can be initiated in *active receive* mode by the allocation
function (see :ref:`start_only_persistent<start_only_persistent_label>`). In this mode,
the persistent worker will always be considered ready to receive more data
(e.g.,~ evaluation results). It can also send to the manager at any time.

The user is responsible for ensuring there are no communication deadlocks
in this mode. Note that in manager/worker message exchanges, only the worker-side
receive is blocking by default (a non-blocking option is available).

Cancelling Simulations
----------------------

Previously submitted simulations can be cancelled by sending a message to the manager.

To do this a PersistentSupport helper function is provided.

.. currentmodule:: libensemble.tools.persistent_support.PersistentSupport
.. autofunction:: request_cancel_sim_ids

If a generated point is cancelled by the generator before it has been given to a
worker for evaluation, then it will never be given. If it has already returned from the
simulation, then results can be returned, but the ``cancel_requested`` field remains
as ``True``. However, if the simulation is running when the manager receives the cancellation
request, a kill signal will be sent to the worker. This can be caught and acted upon
by a user function, otherwise it will be ignored.

The :doc:`Borehole Calibration tutorial<../tutorials/calib_cancel_tutorial>` gives an example
of the capability to cancel pending simulations.

Modification of existing points
-------------------------------

To change existing fields of the history array, the generator can initialize an output
array where the *dtype* contains the ``sim_id`` and the fields to be modified (in
place of ``gen_specs["out"]``), and then send this output array to the manager (as with regular
communications). Any such array received by the manager will overwrite the specific fields
for the given *sim_ids*. If the changes do not correspond with newly generated points,
then the generator needs to communicate to the manager that it is not ready
to receive completed evaluations. Send to the manager with the ``keep_state`` argument
set to *True*.

For example, the cancellation function ``request_cancel_sim_ids`` could be replicated by
the following (where ``sim_ids_to_cancel`` is a list of integers):

.. code-block:: python

    # Send only these fields to existing H rows and libEnsemble will slot in the change.
    H_o = np.zeros(len(sim_ids_to_cancel), dtype=[('sim_id', int), ('cancel_requested', bool)])
    H_o['sim_id'] = sim_ids_to_cancel
    H_o['cancel_requested'] = True
    ps.send(H_o, keep_state=True)

Generator initiated shutdown
----------------------------

If using a supporting allocation function, the generator can prompt the ensemble to shutdown
by simply exiting the function (e.g., on a test for a converged value). For example, the
allocation function :ref:`start_only_persistent<start_only_persistent_label>` closes down
the ensemble as soon a persistent generator returns. The usual return values should be given.

Examples
--------

Examples of non-persistent and persistent generator functions
can be found :doc:`here<../examples/gen_funcs>`.
