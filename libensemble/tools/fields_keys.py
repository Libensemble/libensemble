"""
Below are the fields used within libEnsemble
"""

libE_fields = [('sim_id', int),             # Unique id of entry in H that was generated
               ('gen_worker', int),         # Worker that (first) generated the entry
               ('gen_time', float),         # Time (since epoch) entry (first) was entered into H from a gen
               ('last_gen_time', float),    # Time (since epoch) entry was last requested by a gen
               ('given', bool),             # True if entry has been given for sim eval
               ('given_time', float),       # Time (since epoch) that the entry was (first) given to be evaluated
               ('last_given_time', float),  # Time (since epoch) that the entry was last given to be evaluated
               ('returned', bool),          # True if entry has been returned from sim eval
               ('returned_time', float),    # Time entry was (last) returned from sim eval
               ('sim_worker', int),         # Worker that did (or is doing) the sim eval
               ('cancel_requested', bool),  # True if cancellation of this entry is requested
               ('kill_sent', bool)          # True if a kill signal has been sent to worker
               ]
# end_libE_fields_rst_tag

protected_libE_fields = ['gen_worker',
                         'gen_time',
                         'given',
                         'given_time',
                         'returned',
                         'returned_time',
                         'sim_worker']

allowed_sim_spec_keys = ['sim_f',  #
                         'in',     #
                         'out',    #
                         'user']   #

allowed_gen_spec_keys = ['gen_f',  #
                         'in',     #
                         'out',    #
                         'user']   #

allowed_alloc_spec_keys = ['alloc_f',  #
                           'in',       #
                           'out',      #
                           'user']     #

libE_spec_calc_dir_misc = ['ensemble_copy_back',
                           'ensemble_dir_path',
                           'use_worker_dirs']

libE_spec_sim_dir_keys = ['sim_dirs_make',
                          'sim_dir_copy_files',
                          'sim_dir_symlink_files',
                          'sim_input_dir']

libE_spec_gen_dir_keys = ['gen_dirs_make',
                          'gen_dir_copy_files',
                          'gen_dir_symlink_files',
                          'gen_input_dir']

libE_spec_calc_dir_combined = libE_spec_calc_dir_misc + \
    libE_spec_sim_dir_keys + \
    libE_spec_gen_dir_keys

allowed_libE_spec_keys = ['abort_on_exception',     #
                          'authkey',                #
                          'kill_canceled_sims',    #
                          'comms',                  #
                          'disable_log_files',      #
                          'final_fields',           #
                          'ip',                     #
                          'mpi_comm',               #
                          'nworkers',               #
                          'port',                   #
                          'profile',                #
                          'safe_mode',              #
                          'save_every_k_gens',      #
                          'save_every_k_sims',      #
                          'save_H_and_persis_on_abort',        #
                          'use_persis_return',      #
                          'workerID',               #
                          'worker_timeout',         #
                          'zero_resource_workers',  #
                          'worker_cmd'] + libE_spec_calc_dir_combined
