User Function API
-----------------
.. _user_api:

libEnsemble requires functions for generation, simulation, and allocation.

While libEnsemble provides a default allocation function, the simulator and generator functions
must be specified. The required API and example arguments are given here.
:doc:`Example sim and gen functions<examples/examples_index>` are provided in the
libEnsemble package.

:doc:`See here for more in-depth guides to writing user functions<../function_guides/function_guide_index>`

sim_f API
~~~~~~~~~
.. _api_sim_f:

The simulator function will be called by libEnsemble's workers with the following API::

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
  with keys/value-sizes matching those in sim_specs['out']
  :doc:`(example)<data_structures/history_array>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`

  **calc_status**: :obj:`int`, optional
  Provides a task status to the manager and the libE_stats.txt file
  :doc:`(example)<data_structures/calc_status>`

gen_f API
~~~~~~~~~
.. _api_gen_f:

The generator function will be called by libEnsemble's workers with the following API::

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
  with keys/value-sizes matching those in gen_specs['out']
  :doc:`(example)<data_structures/history_array>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`

  **calc_status**: :obj:`int`, optional
  Provides a task status to the manager and the libE_stats.txt file
  :doc:`(example)<data_structures/calc_status>`

alloc_f API
~~~~~~~~~~~
.. _api_alloc_f:

The allocation function will be called by libEnsemble's manager with the following API::

  Work, persis_info, stop_flag = alloc_f(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info)

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

  **libE_info**: :obj:`dict`
  Various statistics useful to the allocation function for determining how much
  work has been evaluated, or if the routine should prepare to complete. See
  the :doc:`allocation function guide<function_guides/allocator>` for more
  information.

Returns:
********

  **Work**: :obj:`dict`
  Dictionary with integer keys ``i`` for work to be sent to worker ``i``.
  :doc:`(example)<data_structures/work_dict>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<data_structures/persis_info>`

  **stop_flag**: :obj:`int`, optional
  Set to 1 to request libEnsemble manager to stop giving additional work after
  receiving existing work
