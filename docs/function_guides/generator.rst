Generator Functions
===================

As described in the :ref:`API<api_gen_f>`, the ``gen_f`` is called by a
libEnsemble worker via the following::

    out = gen_f(H[gen_specs['in']][sim_ids_from_allocf], persis_info, gen_specs, libE_info)

In practice, most ``gen_f`` function definitions written by users resemble::

    def my_generator(H, persis_info, gen_specs, libE_info):

Where :doc:`H<../data_structures/history_array>` is a selection of the
:doc:`History array<../history_output>`, determined by sim IDs from the
``alloc_f``, :doc:`persis_info<../data_structures/persis_info>` is a dictionary
containing state information, :doc:`gen_specs<../data_structures/gen_specs>` is a
dictionary containing pre-defined parameters for the ``gen_f``, and ``libE_info``
is a dictionary containing libEnsemble-specific entries. See the API above for
more detailed descriptions of the parameters.

.. note::

    If the ``gen_f`` is a persistent generator, then ``gen_specs['in']`` will often be
    empty if the ``alloc_f`` determines what fields to send to the generator.

Typically users start by extracting their custom parameters initially defined
within ``gen_specs['user']`` in the calling script and defining a *local* History
array based on the datatype in ``gen_specs['out']``, to be returned. For example::

        batch_size = gen_specs['user']['batch_size']
        local_H_out = np.zeros(batch_size, dtype=gen_specs['out'])

This array should be populated by whatever values are generated within
the function. Finally, this array should be returned to libEnsemble
alongside ``persis_info``::

        return local_H_out, persis_info

.. note::

    State ``gen_f`` information like checkpointing should be
    appended to ``persis_info``.

.. _persistent-gens:

Persistent Generators
---------------------

While normal generators return after completing their calculation, persistent
generators receive Work units, perform computations, and communicate results
directly to the manager in a loop, not returning until explicitly instructed by
the manager. The calling worker becomes a dedicated :ref:`persistent worker<persis_worker>`.
A ``gen_f`` is initiated as persistent by the ``alloc_f``, which also determines
which structures are sent to the ``gen_f``. In such cases, ``gen_specs`` is often
empty.

Many users prefer persistent generators since they do not need to be
re-initialized every time their past work is completed and evaluated by a
simulation, and an can evaluate returned simulation results over the course of
an entire libEnsemble routine as a single function instance.

Functions for a persistent generator to communicate directly with the manager
are available in the :ref:`libensemble.tools.gen_support<p_gen_routines>` module.
Additional necessary resources are the status tags ``STOP_TAG``, ``PERSIS_STOP``, and
``FINISHED_PERSISTENT_GEN_TAG`` from ``libensemble.message_numbers``, with return
values from the ``gen_support`` functions compared to these tags to determine when
the generator should break its loop and return.

Implementing the above functions is relatively simple:

.. currentmodule:: libensemble.tools.gen_support
.. autofunction:: send_mgr_worker_msg

This function call typically resembles::

    send_mgr_worker_msg(libE_info['comm'], local_H_out[selected_IDs])

Note that ``send_mgr_worker_msg()`` has no return.

.. currentmodule:: libensemble.tools.gen_support
.. autofunction:: get_mgr_worker_msg

This function call typically resembles::

    tag, Work, calc_in = get_mgr_worker_msg(libE_info['comm'])

    if tag in [STOP_TAG, PERSIS_STOP]:
        cleanup()
        break

The logic following the function call is typically used to break the persistent
generator's main loop and return.

.. currentmodule:: libensemble.tools.gen_support
.. autofunction:: sendrecv_mgr_worker_msg

This function performs both of the previous functions in a single statement. Its
usage typically resembles::

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], local_H_out[selected_IDs])
    if tag in [STOP_TAG, PERSIS_STOP]:
        cleanup()
        break

Once the persistent generator's loop has been broken because of
the tag from the manager, it should return with an additional tag::

    return local_H_out, persis_info, FINISHED_PERSISTENT_GEN_TAG

See :doc:`calc_status<../data_structures/calc_status>` for more information about
the message tags.

Examples of normal and persistent generator functions
can be found :doc:`here<../examples/gen_funcs>`.
