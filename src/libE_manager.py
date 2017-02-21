"""
    import IPython
    IPython.embed()
libEnsemble manager routines
====================================================


"""

from __future__ import division
from __future__ import absolute_import

from message_numbers import EVAL_TAG # tell worker to evaluate the point 
from message_numbers import STOP_TAG # tell worker run is over

import numpy as np
import time

def manager_main(comm, history, allocation_specs, sim_specs,
        failure_processing, exit_criteria):

    H, H_ind = initiate_H(sim_specs, exit_criteria)

    # Start out by giving all lead workers a point to evaluate
    for w in allocation_specs['lead_worker_ranks']:
        x_new = sim_specs['gen_f'](sim_specs['gen_f_params'])
        H_ind, rand_ind = send_to_workers_and_update(comm, w, x_new, H, H_ind)

    # while termination_test(H, H_ind, exit_criteria):

    

def send_to_workers_and_update(comm, w, x_new, H, H_ind):
    """
    Set up data and send to worker (w).
    """
    x_new = np.atleast_2d(x_new)
    [batch_size,n] = x_new.shape

    # Set up structure to send
    data_to_send = np.zeros(batch_size, dtype=[('x_scaled','float',n),
                                               ('pt_id','int')])
    data_to_send['pt_id'] = range(H_ind, H_ind + batch_size)
    data_to_send['x_true'] = x_new

    # comm.send(obj=data_to_send, dest=w, tag=EVAL_TAG)

    for i in range(0,batch_size):
        update_history_x(H, H_ind, x_new[i], w)
        H_ind += 1
    
    return H_ind


def update_history_x(H, H_ind, x, lead_rank):
    """
    Updates the history (in place) when a new point has been given to be evaluated

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    H_ind: integer
        The new point
    x: numpy array
        Point to be evaluated
    lead_rank: int
        lead ranks for the evaluation of x 
    """

    H['x_true'][H_ind] = x
    H['pt_id'][H_ind] = H_ind
    H['given'][H_ind] = True
    H['ranks'][H_ind] = lead_rank


def termination_test(H, H_ind, exit_criteria):
    """
    Return False if the libEnsemble run should stop 
    """

    if (H_ind >= exit_criteria['sim_eval_max'] or 
            min(H['f']) <= exit_criteria['min_sim_f_val'] or 
            time.time() - H['given_time'][0] > exit_criteria['elapsed_clock_time']):
        return False
    else:
        return True



def initiate_H(sim_specs, exit_criteria):
    """
    Forms the numpy structured array that records everything from the
    libEnsemble run 

    Returns
    ----------
    H: numpy structured array
        History array storing rows for each point. Field names are below

        | x_scaled            : Parameter values in the unit cube (if bounds)
        | x_true              : Parameter values in the original domain
        | pt_id               : Count of each each point
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
    H['x_true'] = np.inf
    H['given_time'] = np.inf
    H['grad'] = np.nan

    return (H, 0)
