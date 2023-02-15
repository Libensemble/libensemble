User Function API
-----------------
.. _user_api:

libEnsemble requires functions for generation, simulation, and allocation.

While libEnsemble provides a default allocation function, the simulator and generator functions
must be specified. The required API and example arguments are given here.
:doc:`Example sim and gen functions<../examples/examples_index>` are provided in the
libEnsemble package.

:doc:`See here for more in-depth guides to writing user functions<function_guide_index>`

As of v0.9.3+dev, valid simulator and generator functions
can *accept and return a smaller subset of the listed parameters and return values*. For instance,
a ``def my_simulation(one_Input) -> one_Output`` function is now accepted,
as is ``def my_generator(Input, persis_info) -> Output, persis_info``.

sim_f API
~~~~~~~~~
.. _api_sim_f:

The simulator function will be called by libEnsemble's workers with *up to* the following arguments and returns::

    Out, persis_info, calc_status = sim_f(H[sim_specs["in"]][sim_ids_from_allocf], persis_info, sim_specs, libE_info)

Parameters:
***********

  **H**: :obj:`numpy structured array`
  :ref:`(example)<funcguides-history>`

  **persis_info**: :obj:`dict`
  :ref:`(example)<datastruct-persis-info>`

  **sim_specs**: :obj:`dict`
  :ref:`(example)<datastruct-sim-specs>`

  **libE_info**: :obj:`dict`
  :ref:`(example)<funcguides-workdict>`

Returns:
********

  **H**: :obj:`numpy structured array`
  with keys/value-sizes matching those in sim_specs["out"]
  :ref:`(example)<funcguides-history>`

  **persis_info**: :obj:`dict`
  :ref:`(example)<datastruct-persis-info>`

  **calc_status**: :obj:`int`, optional
  Provides a task status to the manager and the libE_stats.txt file
  :ref:`(example)<funcguides-calcstatus>`

gen_f API
~~~~~~~~~
.. _api_gen_f:

The generator function will be called by libEnsemble's workers with *up to* the following arguments and returns::

    Out, persis_info, calc_status = gen_f(H[gen_specs["in"]][sim_ids_from_allocf], persis_info, gen_specs, libE_info)

Parameters:
***********

  **H**: :obj:`numpy structured array`
  :ref:`(example)<funcguides-history>`

  **persis_info**: :obj:`dict`
  :ref:`(example)<datastruct-persis-info>`

  **gen_specs**: :obj:`dict`
  :ref:`(example)<datastruct-gen-specs>`

  **libE_info**: :obj:`dict`
  :ref:`(example)<funcguides-workdict>`

Returns:
********

  **H**: :obj:`numpy structured array`
  with keys/value-sizes matching those in gen_specs["out"]
  :ref:`(example)<funcguides-history>`

  **persis_info**: :obj:`dict`
  :ref:`(example)<datastruct-persis-info>`

  **calc_status**: :obj:`int`, optional
  Provides a task status to the manager and the libE_stats.txt file
  :ref:`(example)<funcguides-calcstatus>`

alloc_f API
~~~~~~~~~~~
.. _api_alloc_f:

The allocation function will be called by libEnsemble's manager with the following API::

  Work, persis_info, stop_flag = alloc_f(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info)

Parameters:
***********

  **W**: :obj:`numpy structured array`
  :doc:`(example)<worker_array>`

  **H**: :obj:`numpy structured array`
  :ref:`(example)<funcguides-history>`

  **sim_specs**: :obj:`dict`
  :ref:`(example)<datastruct-sim-specs>`

  **gen_specs**: :obj:`dict`
  :ref:`(example)<datastruct-gen-specs>`

  **alloc_specs**: :obj:`dict`
  :ref:`(example)<datastruct-alloc-specs>`

  **persis_info**: :obj:`dict`
  :ref:`(example)<datastruct-persis-info>`

  **libE_info**: :obj:`dict`
  Various statistics useful to the allocation function for determining how much
  work has been evaluated, or if the routine should prepare to complete. See
  the :doc:`allocation function guide<allocator>` for more
  information.

Returns:
********

  **Work**: :obj:`dict`
  Dictionary with integer keys ``i`` for work to be sent to worker ``i``.
  :ref:`(example)<funcguides-workdict>`

  **persis_info**: :obj:`dict`
  :doc:`(example)<../data_structures/persis_info>`

  **stop_flag**: :obj:`int`, optional
  Set to 1 to request libEnsemble manager to stop giving additional work after
  receiving existing work
