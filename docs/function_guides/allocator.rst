.. _funcguides-alloc:

Allocation Functions
====================

Although the included allocation functions are sufficient for
most users, those who want to fine-tune how data or resources
may be allocated to their generator or simulator can write their own.

We encourage experimenting with:

1. Prioritization of simulations
2. Sending results immediately or in batch
3. Assigning varying resources to evaluations

.. dropdown:: Example

    ..  literalinclude:: ../../libensemble/alloc_funcs/fast_alloc.py
        :caption: libensemble.alloc_funcs.fast_alloc.give_sim_work_first

The ``alloc_f`` function definition resembles::

    def my_allocator(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):

where:

    * :ref:`W<funcguides-workerarray>` is an array containing worker state info
    * :ref:`H<funcguides-history>` is the *trimmed* History array, containing rows from the generator
    * :ref:`libE_info<libE_info_alloc>` is a set of statistics to determine the progress of work or exit conditions

Most users first check that it is appropriate to allocate work::

        if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
            return {}, persis_info

If the allocation is to continue, instantiate a support class to assist with the
:ref:`Work dictionary<funcguides-workdict>` construction::

        manage_resources = "resource_sets" in H.dtype.names or libE_info["use_resource_sets"]
        support = AllocSupport(W, manage_resources, persis_info, libE_info)
        Work = {}

The Work dictionary is populated with integer keys ``wid`` for each worker and
dictionary values to give to those workers:

.. dropdown:: Example ``Work``

    .. code-block::

        {
            1: {
                "H_fields": ["x"],
                "persis_info": {"rand_stream": RandomState(...) at ..., "worker_num": 1},
                "tag": 1,
                "libE_info": {"H_rows": array([368])}
            },

            2: {
                "H_fields": ["x"],
                "persis_info": {"rand_stream": RandomState(...) at ..., "worker_num": 2},
                "tag": 1,
                "libE_info": {"H_rows": array([369])}
            },

            3: {
                "H_fields": ["x"],
                "persis_info": {"rand_stream": RandomState(...) at ..., "worker_num": 3},
                "tag": 1,
                "libE_info": {"H_rows": array([370])}
            },
            ...

        }

This Work dictionary instructs each worker to call the ``sim_f`` (``tag: 1``)
with data from ``"x"`` and a given ``"H_row"`` from the
History array. A worker-specific ``persis_info`` is also given.

Constructing these arrays and determining which workers are available
for receiving data is simplified by the ``AllocSupport`` class
available within the ``libensemble.tools.alloc_support`` module:

.. dropdown:: AllocSupport

    .. currentmodule:: libensemble.tools.alloc_support
    .. autoclass:: AllocSupport
        :member-order: bysource
        :members:

        .. automethod:: __init__

The Work dictionary is returned to the manager alongside ``persis_info``. If ``1``
is returned as the third value, this instructs the ensemble to stop.

.. note:: An error occurs when the ``alloc_f`` returns nothing while
          all workers are idle

.. _libE_info_alloc:

Information from the manager describing the progress of the current libEnsemble
routine can be found in ``libE_info``::

    libE_info =  {
            "any_idle_workers": bool,            # True if there are any idle workers
            "exit_criteria": {...},              # Criteria for ending routine
            "elapsed_time": float,               # Time elapsed since start of routine
            "gen_informed_count": int,           # Total number of evaluated points given back to a generator function
            "manager_kill_canceled_sims": bool,  # True if manager is to send kills to cancelled simulations
            "scheduler_opts": {...},             # Options passed to the scheduler. "split2fit" and "match_slots"
            "sim_started_count": int,            # Total number of points given for simulation function evaluation
            "sim_ended_count": int,              # Total number of points returned from simulation function evaluations
            "sim_max_given": bool,               # True if `sim_max` simulations have been given out to workers
            "use_resource_sets": bool,           # True if num_resource_sets has been explicitly set.
            "gen_num_procs": int,                # Number of processes used for generator function evaluations
            "gen_num_gpus": int,                 # Number of GPUs used for generator function evaluations
            "gen_on_worker": bool}               # True if generator function is running on a worker

Most often, the allocation function will just return once ``sim_max_given`` is ``True``,
but the user could choose to do something different,
such as cancel points or keep returning completed points to the generator.

Generators that construct models based
on *all evaluated points*, for example, may need simulation work units at the end
of an ensemble to be returned to the generator anyway.

Alternatively, users can use ``elapsed_time`` to track runtime inside their
allocation function and detect impending timeouts, then pack up cleanup work requests,
or mark points for cancellation.

The remaining values above are useful for efficient filtering of H values
(e.g., ``sim_ended_count`` saves filtering by an entire column of H.)

The default allocation function is
``start_only_persistent``. During its worker ID loop, it checks if there's unallocated
work and assigns simulations for that work. Otherwise, it initializes
generators for up to ``"num_active_gens"`` instances. Other settings like
``batch_mode`` are also supported. See
:ref:`here<start_only_persistent_label>` for more information.

.. _examples-alloc:

Examples
========

Below are example allocation functions available in libEnsemble.

Many users use these unmodified.

.. IMPORTANT::
  The default allocation function changed in libEnsemble v2.0 from ``give_sim_work_first`` to ``start_only_persistent``.

.. note::

   The most commonly used allocation function for non-persistent generators is :ref:`give_sim_work_first<gswf_label>`.

.. role:: underline
    :class: underline

.. _start_only_persistent_label:

start_only_persistent
---------------------
.. automodule:: start_only_persistent
  :members:
  :undoc-members:

.. dropdown:: :underline:`start_only_persistent.py`

   .. literalinclude:: ../../libensemble/alloc_funcs/start_only_persistent.py
      :language: python
      :linenos:

.. _gswf_label:

give_sim_work_first
-------------------
.. automodule:: give_sim_work_first
  :members:
  :undoc-members:

.. dropdown:: :underline:`give_sim_work_first.py`

   .. literalinclude:: ../../libensemble/alloc_funcs/give_sim_work_first.py
      :language: python
      :linenos:

fast_alloc
----------
.. automodule:: fast_alloc
  :members:
  :undoc-members:

.. dropdown:: :underline:`fast_alloc.py`

   .. literalinclude:: ../../libensemble/alloc_funcs/fast_alloc.py
      :language: python
      :linenos:

start_persistent_local_opt_gens
-------------------------------
.. automodule:: start_persistent_local_opt_gens
  :members:
  :undoc-members:

fast_alloc_and_pausing
----------------------
.. automodule:: fast_alloc_and_pausing
   :members:
   :undoc-members:

only_one_gen_alloc
------------------
.. automodule:: only_one_gen_alloc
   :members:
   :undoc-members:

start_fd_persistent
-------------------
.. automodule:: start_fd_persistent
   :members:
   :undoc-members:

persistent_aposmm_alloc
-----------------------
.. automodule:: persistent_aposmm_alloc
   :members:
   :undoc-members:

give_pregenerated_work
----------------------
.. automodule:: give_pregenerated_work
   :members:
   :undoc-members:

inverse_bayes_allocf
--------------------
.. automodule:: inverse_bayes_allocf
   :members:
   :undoc-members:
