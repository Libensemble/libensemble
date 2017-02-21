"""
    import IPython
    IPython.embed()
libEnsemble manager routines
====================================================


"""

from __future__ import division
from __future__ import absolute_import

import numpy as np

def manager_main(comm, history, allocation_specs, sim_specs,
        failure_processing, exit_criteria):

    H = initiate_H(sim_specs, exit_criteria)


def initiate_H(sim_specs, exit_criteria):
    """
    Forms the numpy structured array that records everything from the
    libEnsemble run 

    Returns
    ----------
    H: numpy structured array
        History array storing rows for each point. Field names are below

        | x_scaled            : Point in the unit cube
        | x_true              : Point in the original domain
        | pt_id               : Unique ID number for each point
        | local_pt            : True if point is LocalOpt point                  
        | given_time          : Time point was given to a worker
        | lead_rank           : lead worker rank point was given to 
        | returned            : True if point has been evaluated by a worker
        | f                   : Combined value returned from simulation          
        | f_vec               : Vector of function values from simulation        
        | grad                : Gradient (NaN if no derivative returned)
        | worker_start_time   : Evaluation start time                            
        | worker_end_time     : Evaluation end time                              

        | dist_to_unit_bounds : Distance to the boundary
        | dist_to_better_l    : Distance to closest local point with smaller f
        | dist_to_better_s    : Distance to closest sample point with smaller f
        | ind_of_better_l     : Index in H with the closest better local point  
        | ind_of_better_s     : Index in H with the closest better sample point 

        | started_run         : True if point started a LocalOpt run               
        | active              : True if point's run is actively being improved
        | local_min           : True if point was ruled a local min                
                             
    """
    n = sim_specs['sim_f_params']['n']
    m = sim_specs['sim_f_params']['m']
    feval_max = exit_criteria['sim_eval_max']

    H = np.zeros(feval_max, dtype=[('x_scaled','float',n), # initial 
                          ('x_true','float',n),            # initial
                          ('pt_id','int'),                 # initial
                          ('local_pt','bool'),             # initial
                          ('given_time','float'),          # when given
                          ('lead_rank','int'),             # when given
                          ('returned','bool'),             # after eval
                          ('f','float'),                   # after eval  
                          ('f_vec','float',m),             # after eval
                          ('grad','float',n),              # after eval
                          ('worker_start_time','float'),   # after eval
                          ('worker_end_time','float'),     # after eval
                          ('dist_to_unit_bounds','float'), # after eval
                          ('dist_to_better_l','float'),    # after eval
                          ('dist_to_better_s','float'),    # after eval
                          ('ind_of_better_l','int'),       # after eval
                          ('ind_of_better_s','int'),       # after eval
                          ('started_run','bool'),          # run start
                          ('active','bool'),               # during run
                          ('local_min','bool')])           # after run 

    H['dist_to_unit_bounds'] = np.inf
    H['dist_to_better_l'] = np.inf
    H['dist_to_better_s'] = np.inf
    H['ind_of_better_l'] = -1
    H['ind_of_better_s'] = -1
    H['pt_id'] = -1
    H['x_scaled'] = np.inf
    H['grad'] = np.nan

    return H
