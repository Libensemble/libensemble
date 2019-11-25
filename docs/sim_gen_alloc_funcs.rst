User Function API
-----------------

libEnsemble requires functions for generation, simulation and allocation.

While libEnsemble provides a default allocation function, the sim and gen functions
must be provided. The required API and examples are given here.

**The libEnsemble History Array**

libEnsemble uses a NumPy structured array :ref:`H<datastruct-history-array>` to
store output from ``gen_f`` and corresponding ``sim_f`` output. Similarly,
``gen_f`` and ``sim_f`` are expected to return output in NumPy structured
arrays. The names of the fields to be given as input to ``gen_f`` and ``sim_f``
must be an output from ``gen_f`` or ``sim_f``. In addition to the fields output
from ``sim_f`` and ``gen_f``, the final history returned from libEnsemble will
include the following fields:

* ``sim_id`` [int]: Each unit of work output from ``gen_f`` must have an
  associated ``sim_id``. The generator can assign this, but users must be
  careful to ensure points are added in order. For example, ``if alloc_f``
  allows for two ``gen_f`` instances to be running simultaneously, ``alloc_f``
  should ensure that both don’t generate points with the same ``sim_id``.

* ``given`` [bool]: Has this ``gen_f`` output been given to a libEnsemble
  worker to be evaluated yet?

* ``given_time`` [float]: At what time (since the epoch) was this ``gen_f``
  output given to a worker?

* ``sim_worker`` [int]: libEnsemble worker that it was given to be evaluated.

* ``gen_worker`` [int]: libEnsemble worker that generated this ``sim_id``

* ``gen_time`` [float]: At what time (since the epoch) was this entry (or
  collection of entries) put into ``H`` by the manager

* ``returned`` [bool]: Has this worker completed the evaluation of this unit of
  work?

sim_f API
~~~~~~~~~
.. _api_sim_f:

The sim_f function will be called by libEnsemble with the following API::

    out = sim_f(H[sim_specs['in']][sim_ids_from_allocf], persis_info, sim_specs, libE_info)

Parameters:
***********

  **H**: :obj:`numpy structured array`
  :doc:`(example)<data_structures/history_array>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`

  **sim_specs**: :obj:`dict`
  :doc:`(example)<data_structures/sim_specs>`

  **libE_info**: :obj:`dict`
  :doc:`(example)<data_structures/work_dict>`

Returns:
********

  **H**: :obj:`numpy structured array`
  with keys/value-sizes matching those in sim_specs['out'].
  :doc:`(example)<data_structures/history_array>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`

  **calc_status**: :obj:`int`, optional.
  Provides a job status to the Manager and the libE_stats.txt file.
  :doc:`(example)<data_structures/calc_status>`

gen_f API
~~~~~~~~~
.. _api_gen_f:

The gen_f calculations will be called by libEnsemble with the following API::

    out = gen_f(H[gen_specs['in']][sim_ids_from_allocf], persis_info, gen_specs, libE_info)

Parameters:
***********

  **H**: :obj:`numpy structured array`
  :doc:`(example)<data_structures/history_array>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`

  **gen_specs**: :obj:`dict`
  :doc:`(example)<data_structures/gen_specs>`

  **libE_info**: :obj:`dict`
  :doc:`(example)<data_structures/work_dict>`

Returns:
********

  **H**: :obj:`numpy structured array`
  with keys/value-sizes matching those in gen_specs['out'].
  :doc:`(example)<data_structures/history_array>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`

  **calc_status**: :obj:`int`, optional.
  Provides a job status to the Manager and the libE_stats.txt file.
  :doc:`(example)<data_structures/calc_status>`

alloc_f API
~~~~~~~~~~~
.. _api_alloc_f:

The alloc_f calculations will be called by libEnsemble with the following API::

  Work, persis_info = alloc_f(W, H, sim_specs, gen_specs, alloc_specs, persis_info)

Parameters:
***********

  **W**: :obj:`numpy structured array`
  :doc:`(example)<data_structures/worker_array>`

  **H**: :obj:`numpy structured array`
  :doc:`(example)<data_structures/history_array>`

  **sim_specs**: :obj:`dict`
  :doc:`(example)<data_structures/sim_specs>`

  **gen_specs**: :obj:`dict`
  :doc:`(example)<data_structures/gen_specs>`

  **alloc_specs**: :obj:`dict`
  :doc:`(example)<data_structures/alloc_specs>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`

Returns:
********

  **Work**: :obj:`dict`
  Dictionary with integer keys ``i`` for work to be send to worker ``i``.
  :doc:`(example)<data_structures/work_dict>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`
