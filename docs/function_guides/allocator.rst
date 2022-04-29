Allocation Functions
====================

Although the included allocation functions, or ``alloc_f``'s are sufficient for
most users, those who want to fine-tune how data or resources are allocated to their ``gen_f``
and ``sim_f`` can write their own. The ``alloc_f`` is unique since it is called
by libEnsemble's manager instead of a worker.

Most ``alloc_f`` function definitions written by users resemble::

    def my_allocator(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):

Where :doc:`W<../data_structures/worker_array>` is an array containing information
about each worker's state, :doc:`H<../data_structures/history_array>` is the *trimmed* History array,
containing rows initialized by the generator, and ``libE_info`` is a set of selected
statistics for allocation functions to better determine the progress of work or
if various exit conditions have been reached.

Inside an ``alloc_f``, most users first check that it's appropriate to allocate work,
since if all workers are busy or the maximum amount of work has been evaluated, proceeding
with the allocation function is usually not necessary::

        if libE_info['sim_max_given'] or not libE_info['any_idle_workers']:
            return {}, persis_info

If allocation is to continue, a support class is instantiated (see below), and a
:doc:`Work dictionary<../data_structures/work_dict>` is initialized::

        manage_resources = 'resource_sets' in H.dtype.names or libE_info['use_resource_sets']
        support = AllocSupport(W, manage_resources, persis_info, libE_info)
        Work = {}

This Work dictionary is populated with integer keys ``wid`` for each worker and
dictionary values to give to those workers. An example Work dictionary from a run of
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
        ...

    }

Based on information from the API reference above, this Work dictionary
describes instructions for each of the workers to call the ``sim_f`` (``tag: 1``)
with data from the ``'x'`` field and a given ``'H_row'`` from the
History array. A worker specific ``persis_info`` is also given.

Constructing these arrays and determining which workers are available
for receiving data is simplified by use of the ``AllocSupport`` class
available within the ``libensemble.tools.alloc_support`` module:

.. currentmodule:: libensemble.tools.alloc_support
.. autoclass:: AllocSupport
  :member-order: bysource
  :members:

  .. automethod:: __init__

The Work dictionary is returned to the manager alongside ``persis_info``. If ``1``
is returned as third value, this instructs the ensemble to stop.

For allocation functions, as with the other user functions, the level of complexity can
vary widely. Various scheduling and work distribution features are available in
the existing allocation functions, including prioritization of simulations,
returning evaluation outputs to the generator immediately or in batch, assigning
varying resource sets to evaluations, and other methods of fine-tuned control over
the data available to other user functions.

Information from the manager describing the progress of the current libEnsemble
routine can be found in ``libE_info``, passed into the allocation function::

    libE_info =  {'exit_criteria': dict,               # Criteria for ending routine
                  'elapsed_time': float,               # Time elapsed since start of routine
                  'manager_kill_canceled_sims': bool,  # True if manager is to send kills to cancelled simulations
                  'sim_started_count': int,              # Total number of points given for simulation function evaluation
                  'sim_ended_count': int,                # Total number of points returned from simulation function evaluations
                  'gen_informed_count': int,           # Total number of evaluated points given back to a generator function
                  'sim_max_given': bool,               # True if `sim_max` simulations have been given out to workers
                  'use_resource_sets': bool}           # True if num_resource_sets has been explicitly set.

In most supplied examples, the allocation function will just return once ``sim_max_given``
is ``True``, but the user could choose to do something different, such as cancel points or
keep returning completed points to the generator. Generators that construct models based
on *all evaluated points*, for example, may need simulation work units at the end
of an ensemble to be returned to the generator anyway.

Alternatively, users can use ``elapsed_time`` to track the ensembles runtime inside their
allocation function and detect impending timeouts, then pack up cleanup work requests,
or mark points for cancellation.

The remaining values above are useful for efficient filtering of H values
(e.g. ``sim_ended_count``), saves a filtering an entire column of H.

.. note:: An error occurs when the ``alloc_f`` returns nothing while
          all workers are idle

Descriptions of included allocation functions can be found :doc:`here<../examples/alloc_funcs>`.
The default allocation function used by libEnsemble if one isn't specified is
``give_sim_work_first``. During its worker ID loop, it checks if there's unallocated
work and assigns simulations for that work. Otherwise, it initializes
generators for up to ``'num_active_gens'`` instances. Other settings like
``batch_mode`` is also supported. See
:ref:`here<gswf_label>` for more information about ``give_sim_work_first``.

For a shorter, simpler example, here is the ``fast_alloc`` allocation function:

..  literalinclude:: ../../libensemble/alloc_funcs/fast_alloc.py
    :caption: /libensemble/alloc_funcs/fast_alloc.py
