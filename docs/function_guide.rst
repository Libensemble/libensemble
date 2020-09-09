==================================
Writing libEnsemble User Functions
==================================

libEnsemble coordinates ensembles of calculations performed by three main
functions: a :ref:`Generator Function<api_gen_f>`, a :ref:`Simulator Function<api_sim_f>`,
and an :ref:`Allocation Functions<api_alloc_f>`, or ``gen_f``, ``sim_f``, and
``alloc_f`` respectively. These are all referred to as User Functions. Although
libEnsemble includes several ready-to-use User Functions like
:doc:`APOSMM<examples/aposmm>`, it's expected that most users will write their own.
This guide serves as an overview of both necessary and optional components for
writing different kinds of User Functions, and common development patterns.

Generator Functions
===================

As described in the :ref:`API<api_gen_f>`, the ``gen_f`` is called by a
libEnsemble worker via the following::

    out = gen_f(H[gen_specs['in']][sim_ids_from_allocf], persis_info, gen_specs, libE_info)

In practice, most ``gen_f`` function definitions written by users resemble::

    def my_generator(H, persis_info, gen_specs, libE_info):

Where :doc:`H<data_structures/history_array>` is a selection of the
:doc:`History array<history_output>`, determined by sim IDs from the
``alloc_f``, :doc:`persis_info<data_structures/persis_info>` is a dictionary
containing state information, :doc:`gen_specs<data_structures/gen_specs>` is a
dictionary containing pre-defined parameters for the ``gen_f``, and ``libE_info``
is a dictionary containing libEnsemble-specific entries. See the API above for
more detailed descriptions of the parameters.

Typically users start by parsing their custom parameters initially defined
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

Persistent Generators
---------------------

While normal generators return after completing their calculation, persistent
generators receive Work units, perform computations, and communicate results
directly to the manager in a loop, not returning until explicitly instructed by
the manager. The calling worker becomes a dedicated :ref:`persistent worker<persis_worker>`.
The ``gen_f`` is initiated as persistent by the ``alloc_f``.

Functions for a persistent generator to communicate directly with the manager
are available in the :ref:`libensemble.tools.gen_support<p_gen_routines>` module.
Additional necessary resources are the status tags ``STOP_TAG``, ``PERSIS_STOP``, and
``FINISHED_PERSISTENT_GEN_TAG`` from ``libensemble.message_numbers``, with return
values from the ``gen_support`` functions compared to these tags to determine when
the generator should break its loop and return.

Implementing the above functions is relatively simple.

.. currentmodule:: libensemble.tools.gen_support
.. autofunction:: send_mgr_worker_msg

This function call typically resembles::

    send_mgr_worker_msg(libE_info['comm'], local_H_out[selected_IDs])

.. currentmodule:: libensemble.tools.gen_support
.. autofunction:: get_mgr_worker_msg

This function call typically resembles::

    tag, Work, calc_in = get_mgr_worker_msg(libE_info['comm'])

    if tag in [STOP_TAG, PERSIS_STOP]:
        cleanup()
        break

.. currentmodule:: libensemble.tools.gen_support
.. autofunction:: sendrecv_mgr_worker_msg

This function performs both of the previous functions in a single statement. Its
usage typically resembles::

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], local_H_out[selected_IDs])
    if tag in [STOP_TAG, PERSIS_STOP]:
        cleanup()
        break

Once the persistent generator's loop has been broken, it should return with an additional tag::

    return local_H_out, persis_info, FINISHED_PERSISTENT_GEN_TAG

See :doc:`calc_status<data_structures/calc_status>` for more information about
the message tags.

Examples of normal and persistent generator functions
can be found :doc:`here<examples/gen_funcs>`.

Simulator Functions
===================

As described in the :ref:`API<api_sim_f>`, the ``sim_f`` is called by a
libEnsemble worker via a similar interface to the ``gen_f``::

    out = sim_f(H[sim_specs['in']][sim_ids_from_allocf], persis_info, sim_specs, libE_info)

In practice, most ``sim_f`` function definitions written by users resemble::

    def my_simulator(H, persis_info, sim_specs, libE_info):

Where :doc:`sim_specs<data_structures/sim_specs>` is a
dictionary containing pre-defined parameters for the ``sim_f``, and the other
parameters serve similar purposes to those in the ``gen_f``.

The pattern of setting up a local ``H``, parsing out parameters from
``sim_specs``, performing calculations, and returning the local ``H``
with ``persis_info`` should be familiar::

    batch_size = sim_specs['user']['batch_size']
    local_H_out = np.zeros(batch_size, dtype=sim_specs['out'])

    ... # Perform simulation calculations

    return local_H_out, persis_info

Descriptions of included simulator functions can be found :ref:`here<examples/sim_funcs>`.

The :ref:`Simple Sine tutorial<tutorials/local_sine_tutorial>` is an
excellent introduction for writing simple user functions and using them
with libEnsemble.

Executor
--------

libEnsemble's Executor is commonly used within simulator functions to launch
and monitor applications. An excellent overview is already available
:doc:`here<executor/overview>`.

See the :doc:`Executor with Electrostatic Forces tutorial<tutorials/executor_forces_tutorial>`
for an additional example to try out.

Allocation Functions
====================

Although the included allocation functions, or ``alloc_f``'s are sufficient for
most users, those who want to fine-tune how data is passed to their ``gen_f``
and ``sim_f`` can write their own. The ``alloc_f`` is unique since it is called
by the libEnsemble's manager instead of a worker.

Most ``alloc_f`` function definitions written by users resemble::

    def my_allocator(W, H, sim_specs, gen_specs, alloc_specs, persis_info):

Where :doc:`W<data_structures/worker_array>` is an array containing information
about each worker's state, and ``H`` is the *entire* History array available to
the manager at the point the ``alloc_f`` is called.

Inside an ``alloc_f``, a :doc:`Work dictionary<data_structures/work_dict>` is
instantiated::

    Work = {}

then populated with integer keys ``i`` for each worker and dictionary values to
give to those workers. An example Work dictionary from a run of
the ``test_1d_sampling.py`` regression test resembles::

    {
        1: {
            'H_fields': ['x'],
            'persis_info': {'rand_stream': RandomState(...) at ..., 'worker_num': 1},
            'tag': 1,
            'libE_info': {'H_rows': array([368])}
        },

        2: {
            'H_fields': ['x'],
            'persis_info': {'rand_stream': RandomState(...) at ..., 'worker_num': 2},
            'tag': 1,
            'libE_info': {'H_rows': array([369])}
        },

        3: {
            'H_fields': ['x'],
            'persis_info': {'rand_stream': RandomState(...) at ..., 'worker_num': 3},
            'tag': 1,
            'libE_info': {'H_rows': array([370])}
        },

        4: {
            'H_fields': ['x'],
            'persis_info': {'rand_stream': RandomState(...) at ..., 'worker_num': 4},
            'tag': 1,
            'libE_info': {'H_rows': array([371])}
        }
    }

Based on information from the API reference above, this Work dictionary
describes instructions for each of the four workers to call the ``sim_f``
with data from the ``'x'`` field and a given ``'H_row'`` from the
History array, and also pass ``persis_info``.

Constructing these arrays and determining which workers are available
for receiving data is simplified by several functions available within the
``libensemble.tools.alloc_support`` module:

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: avail_worker_ids

Many ``alloc_f`` routines loop over the available workers returned by the above
function to construct their Work dictionaries with the help of the following two
functions.

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: sim_work

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: gen_work

Note that these two functions *append* an entry in-place to the Work dictionary,
and that additional parameters are appended to ``libE_info``.

In practice, the structure of many allocation functions resemble::

    Work = {}
    ...
    for ID in avail_worker_ids(W):
        ...
        if some_condition:
            sim_work(Work, ID, chosen_H_fields, chosen_H_rows, persis_info)
            ...

        if another_condition:
            gen_work(Work, ID, chosen_H_fields, chosen_H_rows, persis_info)
            ...

    return Work, persis_info

The final three functions available in the ``alloc_support`` module
are primarily for evaluating running generators:

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: test_any_gen

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: count_gens

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: count_persis_gens

Descriptions of included allocation functions can be found :doc:`here<examples/alloc_funcs>`.
Below is the ``fast_alloc`` allocation function as an example:

..  literalinclude:: ../libensemble/alloc_funcs/fast_alloc.py
    :caption: /libensemble/alloc_funcs/fast_alloc.py
