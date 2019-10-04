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
