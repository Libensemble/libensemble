from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first

alloc_specs = {'alloc_f': give_sim_work_first,
               'out': [('allocated', bool)],
               'user': {'batch_mode': True, 'num_active_gens': 1}}
# end_alloc_specs_rst_tag
