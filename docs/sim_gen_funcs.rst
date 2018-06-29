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

    out = gen_f(H[gen_specs['in']][sim_ids_from_allocf], persis_info, sim_specs, libE_info)

again, where out is a tuple containing (H, persis_info, [calc_tag]). H is a numpy structured array with keys/value-sizes matching those in gen_specs['out']. Work, persis_info = alloc_f(worker_sets, H, persis_info, sim_specs, libE_info) where Work[i] is a dictionary of work to be sent to worker i. Work[i] should contain the following fields:
