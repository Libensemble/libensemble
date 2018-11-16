Sim, Gen, and Alloc functions API
=================================

.. _api_sim_f:

sim_f API
---------

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

  **calc_tag**: :obj:`int`, optional
  Used to tell manager why a persistent worker is stopping.

..  literalinclude:: ../libensemble/message_numbers.py
    :lines: 1-8

.. _api_gen_f:

gen_f API
---------

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

  **calc_tag**: :obj:`int`, optional
  Used to tell manager why a persistent worker is stopping.

..  literalinclude:: ../libensemble/message_numbers.py
    :lines: 1-8

.. _api_alloc_f:

alloc_f API
-----------

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

