"""
Below are the fields used within libEnsemble
"""
libE_fields = [('sim_id', int),        # The number of the entry in H that was generated
               ('given', bool),        # True if the entry has been given to a worker for a sim eval
               ('given_time', float),  # Time (since epoch) that the entry was given to a worker for a sim eval
               ('sim_worker', int),    # Worker that did (or is doing) the sim eval
               ('gen_worker', int),    # Worker that generated the entry
               ('gen_time', float),    # Time (since epoch) that entry was entered into H
               ('returned', bool),     # True if the entry has been returned from the sim eval
               ]
