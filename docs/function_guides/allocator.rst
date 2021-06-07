Allocation Functions
====================

Although the included allocation functions, or ``alloc_f``'s are sufficient for
most users, those who want to fine-tune how data or resources are allocated to their ``gen_f``
and ``sim_f`` can write their own. The ``alloc_f`` is unique since it is called
by the libEnsemble's manager instead of a worker.

Most ``alloc_f`` function definitions written by users resemble::

    def my_allocator(W, H, sim_specs, gen_specs, alloc_specs, persis_info):

Where :doc:`W<../data_structures/worker_array>` is an array containing information
about each worker's state, and ``H`` is the *trimmed* History array,
containing rows initialized by the generator.

Inside an ``alloc_f``, a :doc:`Work dictionary<../data_structures/work_dict>` is
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

Note that these two functions *append* an entry in-place to the Work dictionary
and additional parameters are appended to ``libE_info``.

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

The Work dictionary is returned to the manager alongside ``persis_info``. If ``1``
is returned as third value, this instructs the run to stop.

For allocation functions, as with the user functions, the level of complexity can
vary widely. Various scheduling and work distribution features are available in
the existing allocation functions, including prioritization of simulations,
returning evaluation outputs to the generator immediately or in batch, assigning
varying resource sets to evaluations, and other methods of fine-tuned control over
the data available to other user functions.

.. note:: An error occurs when the ``alloc_f`` returns nothing while
          all workers are idle

The final three functions available in the ``alloc_support`` module
are primarily for evaluating running generators:

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: test_any_gen

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: count_gens

.. currentmodule:: libensemble.tools.alloc_support
.. autofunction:: count_persis_gens

Descriptions of included allocation functions can be found :doc:`here<../examples/alloc_funcs>`.
The default allocation function used by libEnsemble if one isn't specified is
``give_sim_work_first``. During its worker ID loop, it checks if there's unallocated
work and assigns simulations for that work if so. Otherwise, it initializes
generators for up to ``'num_active_gens'`` instances. Other settings like
``batch_mode`` and blocking of non-active workers is also supported. See
:ref:`here<gswf_label>` for more information about ``give_sim_work_first``.

For a shorter, simpler example, here is the ``fast_alloc`` allocation function:

..  literalinclude:: ../../libensemble/alloc_funcs/fast_alloc.py
    :caption: /libensemble/alloc_funcs/fast_alloc.py
