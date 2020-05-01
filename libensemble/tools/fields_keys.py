"""
Below are the fields used within libEnsemble
"""

libE_fields = [('sim_id', int),        # Unique id of entry in H that was generated
               ('gen_worker', int),    # Worker that generated the entry
               ('gen_time', float),    # Time (since epoch) entry was entered into H
               ('given', bool),        # True if entry has been given for sim eval
               ('returned', bool),     # True if entry has been returned from sim eval
               ('given_time', float),  # Time (since epoch) that the entry was given
               ('sim_worker', int),    # Worker that did (or is doing) the sim eval
               ]
# end_libE_fields_rst_tag

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

libE_spec_calc_dir_keys = ['sim_dirs_make',
                           'ensemble_copy_back',
                           'sim_dir_copy_files',
                           'ensemble_dir_path',
                           'use_worker_dirs',
                           'sim_dir_symlink_files',
                           'sim_input_dir']

allowed_libE_spec_keys = ['abort_on_exception',     #
                          'authkey',                #
                          'comm',                   #
                          'comms',                  #
                          'disable_log_files',      #
                          'ip',                     #
                          'nworkers',               #
                          'port',                   #
                          'profile_worker',         #
                          'save_every_k_gens',      #
                          'save_every_k_sims',      #
                          'save_H_and_persis_on_abort',        #
                          'workerID',               #
                          'worker_timeout',         #
                          'worker_cmd'] + libE_spec_calc_dir_keys
