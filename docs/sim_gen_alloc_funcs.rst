Sim and Gen Functions
=====================

sim_f API
---------

The sim_f function will be called by libEnsemble with the following API::

    out = sim_f(H[sim_specs['in']][sim_ids_from_allocf], persis_info, sim_specs, libE_info)

where out is a tuple containing (H, persis_info, [calc_tag]). H is a numpy structured array with
keys/value-sizes matching those in sim_specs['out'].

gen_f API
---------

The gen_f calculations will be called by libEnsemble with the following API::

    out = gen_f(H[gen_specs['in']][sim_ids_from_allocf], persis_info, gen_specs, libE_info)

again, where out is a tuple containing (H, persis_info, [calc_tag]). H is a numpy structured array with keys/value-sizes matching those in gen_specs['out']. 

alloc_f API
-----------

The alloc_f calculations will be called by libEnsemble with the following API::

  Work, persis_info = alloc_f(W, H, sim_specs, gen_specs, persis_info)

Parameters:
***********

**W**: :obj:`numpy strucutred array`
:doc:`(example)<data_structures/worker_array>`

**H**: :obj:`numpy strucutred array`
:doc:`(example)<data_structures/history_array>`

**sim_specs**: :obj:`dict`
:doc:`(example)<data_structures/sim_specs>`

**gen_specs**: :obj:`dict`
:doc:`(example)<data_structures/gen_specs>`

**persis_info**: :obj:`dict`
:doc:`(example)<data_structures/persis_info>`


Returns:
********

**Work**: :obj:`dict`
Dictionary with integer keys ``i`` for work to be send to worker ``i``.
:doc:`(example)<data_structures/work_dict>`

**persis_info**: :obj:`dict`
:doc:`(example)<data_structures/persis_info>`

