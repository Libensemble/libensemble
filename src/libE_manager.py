"""
    import IPython
    IPython.embed()
libEnsemble manager routines
====================================================


"""

from __future__ import division
from __future__ import absolute_import

from message_numbers import EVAL_TAG # manager tells worker to evaluate the point 
from message_numbers import STOP_TAG # manager tells worker run is over

from mpi4py import MPI
import numpy as np
import scipy as sp
from scipy import spatial

import time,sys

def manager_main(comm, history, allocation_specs, sim_specs,
        failure_processing, exit_criteria):

    status = MPI.Status()
    sim_params = sim_specs['sim_f_params']

    H, H_ind = initiate_H(sim_specs, exit_criteria)

    ### Start out by giving all lead workers a point to evaluate
    for w in allocation_specs['lead_worker_ranks']:
        x_new = sim_specs['gen_f'](sim_specs['gen_f_params'])
        H_ind = send_to_workers_and_update(comm, w, x_new, H, H_ind, sim_params)

    idle_w = set([])
    active_w = allocation_specs['lead_worker_ranks'].copy()

    ### Continue receiving and giving until termination test is satisfied
    while termination_test(H, H_ind, exit_criteria):
        data_received = comm.recv(buf=None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        if status.Get_source() in active_w:
            idle_w.add(status.Get_source())
            active_w.remove(status.Get_source())
            update_history_f(H, data_received, sim_params['combine_func'])
        else:
            sys.exit('Received from non-worker')

        for w in sorted(idle_w):
            x_new = sim_specs['gen_f'](sim_specs['gen_f_params'])
            H_ind = send_to_workers_and_update(comm, w, x_new, H, H_ind, sim_params)
            active_w.add(w)
            idle_w.remove(w)



    ### Receive from all active workers 
    for w in active_w:
        data_received = comm.recv(buf=None, source=w, tag=MPI.ANY_TAG, status=status)
        update_history_f(H, data_received, sim_params['combine_func'])

    ### Stop all workers 
    for w in allocation_specs['lead_worker_ranks']:
        comm.send(obj=None, dest=w, tag=STOP_TAG)

    return(H[:H_ind])


def update_history_f(H, data_received, combine_func): 
    """
    Updates the history (in place) after a point has been evaluated

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    data['x_scaled']: numpy array
        Point that was evaluated
    data['pt_id']: int
        The pt_id associated with x
    data['f']: float
        Function value
    data['f_vec']: numpy array
        Vector of function values
    data['grad']: numpy array
        Gradient vector
    data['worker_start_time']: float
        Start time of evaluation
    data['worker_end_time']: float
        End time of evaluation
    """

    n = len(data_received[0]['x_true'])

    for data in data_received:
        new_pt = data['pt_id']
        assert (H['x_true'][new_pt] == data['x_true']).all(), \
                "Worker-returned x-value different than History x-value"

        H['returned'][new_pt] = True
        H['f'][new_pt] = combine_func(data['f_vec'])
        H['f_vec'][new_pt] = data['f_vec']
        H['grad'][new_pt] = data['grad']
        H['worker_start_time'][new_pt] = data['worker_start_time']
        H['worker_end_time'][new_pt] = data['worker_end_time']

        # Points with pt_id's 
        p = H['pt_id']>=0

        dist_to_all = sp.spatial.distance.cdist(np.atleast_2d(H['x_scaled'][new_pt]), H['x_scaled'][p], 'euclidean').flatten()
        new_better_than = np.logical_and(H['f'][new_pt] < H['f'][p], H['returned'][p])

        # Compute distance to boundary
        H['dist_to_unit_bounds'][new_pt] = min(min(np.ones(n) - H['x_scaled'][new_pt]),min(H['x_scaled'][new_pt] - np.zeros(n)))

        # Update any other points if new_pt is closer and better
        if H['local_pt'][new_pt]:
            updates = np.where(np.logical_and(dist_to_all < H['dist_to_better_l'][p], new_better_than))[0]
            H['dist_to_better_l'][updates] = dist_to_all[updates]
            H['ind_of_better_l'][updates] = new_pt
        else:
            updates = np.where(np.logical_and(dist_to_all < H['dist_to_better_s'][p], new_better_than))[0]
            H['dist_to_better_s'][updates] = dist_to_all[updates]
            H['ind_of_better_s'][updates] = new_pt

        # If we allow equality, have to prevent new_pt from being its own "better point"
        better_than_new_l = np.logical_and.reduce((H['f'][new_pt] >= H['f'][p],  H['local_pt'][p], H['returned'][p], H['pt_id'][p] != new_pt))
        better_than_new_s = np.logical_and.reduce((H['f'][new_pt] >= H['f'][p], ~H['local_pt'][p], H['returned'][p], H['pt_id'][p] != new_pt))

        # Who is closest to ind and better 
        if np.any(better_than_new_l):
            H['dist_to_better_l'][new_pt] = dist_to_all[better_than_new_l].min()
            H['ind_of_better_l'][new_pt] = np.ix_(better_than_new_l)[0][dist_to_all[better_than_new_l].argmin()]

        if np.any(better_than_new_s):
            H['dist_to_better_s'][new_pt] = dist_to_all[better_than_new_s].min()
            H['ind_of_better_s'][new_pt] = np.ix_(better_than_new_s)[0][dist_to_all[better_than_new_s].argmin()]

    

def send_to_workers_and_update(comm, w, x_new, H, H_ind, sim_f_params):
    """
    Set up data and send to worker (w).
    """
    x_new = np.atleast_2d(x_new)
    batch_size, n = x_new.shape

    # Set up structure to send
    data_to_send = np.zeros(batch_size, dtype=[('x_true','float',n),
                                               ('pt_id','int')])
    data_to_send['pt_id'] = range(H_ind, H_ind + batch_size)
    data_to_send['x_true'] = x_new

    comm.send(obj=data_to_send, dest=w, tag=EVAL_TAG)

    for i in range(0,batch_size):
        update_history_x(H, H_ind, x_new[i], w, sim_f_params)
        H_ind += 1
    
    return H_ind


def update_history_x(H, H_ind, x, lead_rank, sim_f_params):
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

    H['x_scaled'][H_ind] =  (x - sim_f_params['lb'])/(sim_f_params['ub']-sim_f_params['lb'])
    H['x_true'][H_ind] = x
    H['pt_id'][H_ind] = H_ind
    H['local_pt'][H_ind] = False
    H['given'][H_ind] = True
    H['given_time'][H_ind] = time.time()
    H['lead_rank'][H_ind] = lead_rank


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
        | given               : True if point has been given to a worker
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

    H = np.zeros(feval_max, dtype=[('x_scaled','float',n), # when given 
                          ('x_true','float',n),            # when given
                          ('pt_id','int'),                 # when given
                          ('local_pt','bool'),             # when given
                          ('given','bool'),                # when given
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
