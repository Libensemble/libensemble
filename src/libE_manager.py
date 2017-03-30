"""
    import IPython; IPython.embed()
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

def manager_main(comm, allocation_specs, sim_specs, gen_specs,
        failure_processing, exit_criteria):

    status = MPI.Status()

    H, H_ind = initiate_H(sim_specs, gen_specs, exit_criteria)

    idle_w = allocation_specs['worker_ranks'].copy()
    active_w = set([])

    ### Continue receiving and giving until termination test is satisfied
    while termination_test(H, H_ind, exit_criteria):

        active_w, idle_w = receive_from_sim_and_gen(comm, active_w, idle_w, H)

        update_active_and_queue(active_w, idle_w, H)

        Work, H_ind = decide_work_and_resources(active_w, idle_w, H, H_ind, sim_specs, gen_specs)

        for w in Work:
            comm.send(obj=Work[w], dest=w, tag=EVAL_TAG)
            active_w.add(w)
            idle_w.remove(w)

    ### Receive from all active workers 
    for w in active_w:
        data_received = comm.recv(buf=None, source=w, tag=MPI.ANY_TAG, status=status)
        update_history_f(H, data_received, sim_params['combine_func'])

    ### Stop all workers 
    for w in allocation_specs['worker_ranks']:
        comm.send(obj=None, dest=w, tag=STOP_TAG)

    return H[:H_ind]


def receive_from_sim_and_gen(comm, active_w, idle_w, H):
    status = MPI.Status()
    new_stuff = True

    while new_stuff:
        new_stuff = False
        for w in active_w: 
            if comm.Iprobe(source=w, tag=MPI.ANY_TAG):
                D_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
                idle_w.add(status.Get_source())
                active_w.remove(status.Get_source())

                if D_recv['calc_type'] == 'sim':
                    update_history_f(H, data_received)
                elif D_recv['calc_type'] == 'gen':
                    H_ind = update_history_x_in(H, H_ind, D_recv['calc_out'])

                new_stuff = True

    return active_w, idle_w

def decide_work_and_resources(active_w, idle_w, H, H_ind, sim_specs, gen_specs):
    """ Decide what workers should be given
    """

    Work = {}

    for i in idle_w:
        if len(Q):
            v = Q.get_all_from_highest_priority_run()
            Work[i] = {'calc_f': sim_specs['f'], 
                       'calc_params': v[0], 
                       'form_subcomm': [], 
                       'calc_dir': sim_specs['obj_dir'],
                       'calc_type': 'sim',
                       'calc_in': sim_specs['in'],
                       'calc_out': sim_specs['out'],
                      }

            H_ind = update_history_x_out(H, H_ind, v[0], w, sim_specs['params'])

        else:
            Work[i] = {'calc_f': gen_specs['f'], 
                       'calc_params': gen_specs['params'], 
                       'form_subcomm': [], 
                       'calc_dir': '',
                       'calc_type': 'gen',
                       'calc_in': gen_specs['in'],
                       'calc_out': gen_specs['out'],
                       }

    return Work, Q, H_ind

def update_active_and_queue(active_w, idle_w, H, Q):
    """ Decide if active work should be continued and the queue order

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    Q: Queue of points to be evaluated (and their resources)
    """

    return Q

def update_history_f(H, data_received): 
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
        H['f'][new_pt] = data['f']
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

    




def update_history_x_in(H, H_ind, X):
    """
    Updates the history (in place) when a new point has been returned from a gen

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    H_ind: integer
        The new point
    X: numpy array
        Points to be evaluated
    """
    b = len(X)

    H['x'][H_ind:H_ind+b] = O['x']
    H['priority'][H_ind:H_ind+b] = O['priority']

    H_ind += b

    return H_ind

def update_history_x_out(H, H_ind, X, lead_rank, sim_f_params):
    """
    Updates the history (in place) when a new point has been given out to be evaluated

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    H_ind: integer
        The new point
    X: numpy array
        Points to be evaluated
    lead_rank: int
        lead ranks for the evaluation of x 
    """

    for x in np.atleast_2d(X):
        H['x_scaled'][H_ind] =  (x - sim_f_params['lb'])/(sim_f_params['ub']-sim_f_params['lb'])
        H['x_true'][H_ind] = x
        H['pt_id'][H_ind] = H_ind
        H['local_pt'][H_ind] = False
        H['given'][H_ind] = True
        H['given_time'][H_ind] = time.time()
        H['lead_rank'][H_ind] = lead_rank

        H_ind = H_ind + 1

    return H_ind


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



def initiate_H(sim_specs, gen_specs, exit_criteria):
    """
    Forms the numpy structured array that records everything from the
    libEnsemble run 

    Returns
    ----------
    H: numpy structured array
        History array storing rows for each point. Field names are below

        | pt_id               : Count of each each point
        | given               : True if point has been given to a worker
        | given_time          : Time point was given to a worker
        | lead_rank           : lead worker rank point was given to 
        | returned            : True if point has been evaluated by a worker

    """

    default_keys = [('pt_id','int'),
                    ('given','bool'),       
                    ('given_time','float'), 
                    ('lead_rank','int'),    
                    ('returned','bool'),    
                   ]

    feval_max = exit_criteria['sim_eval_max']

    H = np.zeros(feval_max, dtype=default_keys + sim_specs['out'] + gen_specs['out']) 

    H['pt_id'] = -1
    H['given_time'] = np.inf

    return (H, 0)
