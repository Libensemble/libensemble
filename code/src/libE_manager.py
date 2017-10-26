"""
    import IPython; IPython.embed()
    sys.stdout.flush()
    import ipdb; ipdb.set_trace()
    import pdb; pdb.set_trace()
libEnsemble manager routines
====================================================
"""

from __future__ import division
from __future__ import absolute_import

# from message_numbers import EVAL_TAG # manager tells worker to evaluate the point 
from message_numbers import EVAL_SIM_TAG 
from message_numbers import EVAL_GEN_TAG 
from message_numbers import STOP_TAG # manager tells worker run is over

from mpi4py import MPI
import numpy as np

import time, sys, os
import copy

def manager_main(comm, alloc_specs, sim_specs, gen_specs, failure_processing, exit_criteria, H0):

    H, H_ind, term_test, idle_w, active_w = initialize(sim_specs, gen_specs, alloc_specs, exit_criteria, H0)
    persistent_queue_data = {}; gen_info = {}

    send_initial_info_to_workers(comm, H, sim_specs, gen_specs, idle_w)

    ### Continue receiving and giving until termination test is satisfied
    while not term_test(H, H_ind):

        H, H_ind, active_w, idle_w = receive_from_sim_and_gen(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs)

        persistent_queue_data = update_active_and_queue(active_w, idle_w, H[:H_ind], gen_specs, persistent_queue_data)

        Work = alloc_specs['alloc_f'](active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test, gen_info)

        for w in Work:
            active_w, idle_w = send_to_worker_and_update_active_and_idle(comm, H, Work[w], w, sim_specs, gen_specs, active_w, idle_w)

    H, exit_flag = final_receive_and_kill(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test, alloc_specs)

    return H, exit_flag





######################################################################
# Manager subroutines
######################################################################
def send_initial_info_to_workers(comm, H, sim_specs, gen_specs, idle_w):
    # Communicate the gen dtype to workers to save time on future
    # communications. (Must communicate this when workers are requesting
    # libE_fields that aren't in sim_specs['out'] or gen_specs['out'])
    for w in idle_w:
        comm.send(obj=H[sim_specs['in']].dtype, dest=w)
        comm.send(obj=H[gen_specs['in']].dtype, dest=w)

def send_to_worker_and_update_active_and_idle(comm, H, Work, w, sim_specs, gen_specs, active_w, idle_w):

    comm.send(obj=Work['info'], dest=w, tag=Work['tag'])
    if len(Work['info']['H_rows']):
        for i in Work['H_fields']:
            comm.send(obj=H[i][0].dtype,dest=w)
            comm.Send(H[i][Work['info']['H_rows']], dest=w)

    active_w[Work['tag']].add(w)
    idle_w.remove(w)

    if 'blocking' in Work['info']:
        active_w['blocked'].update(Work['info']['blocking'])
        idle_w.difference_update(Work['info']['blocking'])

    return active_w, idle_w


def receive_from_sim_and_gen(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs):

    status = MPI.Status()

    new_stuff = True
    while new_stuff:
        new_stuff = False
        for w in active_w[EVAL_SIM_TAG].copy() | active_w[EVAL_GEN_TAG].copy(): 
            if comm.Iprobe(source=w, tag=MPI.ANY_TAG, status=status):
                D_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
                recv_tag = status.Get_tag()
                idle_w.add(w)
                active_w[recv_tag].remove(w) 
                new_stuff = True

                assert recv_tag in [EVAL_SIM_TAG, EVAL_GEN_TAG], 'Unknown calculation tag received. Exiting'
                if recv_tag == EVAL_SIM_TAG:
                    update_history_f(H, D_recv)
                else: # recv_tag == EVAL_GEN_TAG:
                    H, H_ind = update_history_x_in(H, H_ind, D_recv['calc_out'])

                if 'blocking' in D_recv['info']:
                    active_w['blocked'].difference_update(D_recv['info']['blocking'])
                    idle_w.update(D_recv['info']['blocking'])

    if 'save_every_k' in sim_specs:
        k = sim_specs['save_every_k']
        count = k*(sum(H['returned'])//k)
        filename = 'libE_history_after_sim_' + str(count) + '.npy'

        if not os.path.isfile(filename) and count > 0:
            np.save(filename,H)

    if 'save_every_k' in gen_specs:
        k = gen_specs['save_every_k']
        count = k*(H_ind//k)
        filename = 'libE_history_after_gen_' + str(count) + '.npy'

        if not os.path.isfile(filename) and count > 0:
            np.save(filename,H)

    return H, H_ind, active_w, idle_w


def update_active_and_queue(active_w, idle_w, H, gen_specs, data):
    """ Decide if active work should be continued and the queue order

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    """
    if 'queue_update_function' in gen_specs:
        H, data = gen_specs['queue_update_function'](H,gen_specs, data)
    
    return data


def update_history_f(H, D): 
    """
    Updates the history (in place) after a point has been evaluated

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    """

    new_inds = D['info']['H_rows']
    H_0 = D['calc_out']

    for j,ind in enumerate(new_inds): 
        for field in H_0.dtype.names:
            H[field][ind] = H_0[field][j]

        H['returned'][ind] = True


def update_history_x_in(H, H_ind, O):
    """
    Updates the history (in place) when a new point has been returned from a gen

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    H_ind: integer
        The new point
    O: numpy array
        Output from gen_f
    """

    rows_remaining = len(H)-H_ind
    
    if 'sim_id' not in O.dtype.names:
        # gen method must not be adjusting sim_id, just append to H
        num_new = len(O)

        if num_new > rows_remaining:
            H = grow_H(H,num_new-rows_remaining)
            
        update_inds = np.arange(H_ind,H_ind+num_new)
        H['sim_id'][H_ind:H_ind+num_new] = range(H_ind,H_ind+num_new)
    else:
        # gen method is building sim_id. 
        num_new = len(np.setdiff1d(O['sim_id'],H['sim_id']))

        if num_new > rows_remaining:
            H = grow_H(H,num_new-rows_remaining)

        update_inds = O['sim_id']
        
    for field in O.dtype.names:
        H[field][update_inds] = O[field]

    H_ind += num_new

    return H, H_ind


def grow_H(H, k):
    """ libEnsemble is requesting k rows be added to H because gen_f produced
    more points than rows in H."""
    H_1 = np.zeros(k, dtype=H.dtype)
    H_1['sim_id'] = -1
    H_1['given_time'] = np.inf

    H = np.append(H,H_1)

    return H



def termination_test(H, H_ind, exit_criteria, start_time, lenH0):
    """
    Return nonzero if the libEnsemble run should stop 
    """

    if 'sim_max' in exit_criteria:
        if np.sum(H['given']) >= exit_criteria['sim_max'] + lenH0:
            return 1

    if 'gen_max' in exit_criteria:
        if H_ind >= exit_criteria['gen_max'] + lenH0:
            return 1 

    if 'stop_val' in exit_criteria:
        key = exit_criteria['stop_val'][0]
        val = exit_criteria['stop_val'][1]
        if np.any(H[key][:H_ind][~np.isnan(H[key][:H_ind])] <= val): 
            return 1

    if 'elapsed_wallclock_time' in exit_criteria:
        if time.time() - start_time >= exit_criteria['elapsed_wallclock_time']:
            return 2

    return False


def initialize(sim_specs, gen_specs, alloc_specs, exit_criteria, H0):
    """
    Forms the numpy structured array that records everything from the
    libEnsemble run 

    Returns
    ----------
    H: numpy structured array
        History array storing rows for each point. Field names are below

        | sim_id              : Identifier for each simulation (could be the same "point" just with different parameters) 
        | given               : True if point has been given to a worker
        | given_time          : Time point was given to a worker
        | lead_rank           : lead worker rank point was given to 
        | returned            : True if point has been evaluated by a worker

    H_ind: integer
        Where libEnsemble should start filling in H

    term_test: lambda funciton
        Simplified termination test (doesn't require passing fixed quantities).
        This is nice when calling term_test in multiple places.
    """

    if 'sim_max' in exit_criteria:
        L = exit_criteria['sim_max']
    else:
        L = 100

    libE_fields = [('sim_id',int),
                   ('given',bool),       
                   ('given_time',float), 
                   ('lead_rank',int),    
                   ('returned',bool),    
                   ('paused',bool),    
                   ]

    if ('sim_id',int) in gen_specs['out'] and 'sim_id' in gen_specs['in']:
        print('\n' + 79*'*' + '\n'
               "User generator script will be creating sim_id.\n"\
               "Take care to do this sequentially.\n"\
               "Also, any information given back for existing sim_id values will be overwritten!\n"\
               "So everything in gen_out should be in gen_in!"\
                '\n' + 79*'*' + '\n\n')
        sys.stdout.flush()
        libE_fields = libE_fields[1:] # Must remove 'sim_id' from libE_fields because it's in gen_specs['out']

    H = np.zeros(L + len(H0), dtype=libE_fields + sim_specs['out'] + gen_specs['out']) 

    if len(H0):
        fields = H0.dtype.names
        assert set(fields).issubset(set(H.dtype.names)), "H0 contains fields not in H. Exiting"
        if 'returned' in fields:
            assert np.all(H0['returned']), "H0 contains unreturned points. Exiting"
        if 'obj_component' in fields:
            assert np.max(H0['obj_component']) < gen_specs['components'], "H0 has more obj_components than exist for this problem. Exiting."

        for field in fields:
            assert H[field].ndim == H0[field].ndim, "H0 and H have different ndim for field: " + field + ". Exiting"
            assert np.all(np.array(H[field].shape) >= np.array(H0[field].shape)), "H is not large enough to receive all of the components of H0 in field: " + field + ". Exiting"

            H[field][:len(H0)] = H0[field]
            # for ind, val in np.ndenumerate(H0[field]): # Works if H0[field] has arbitrary dimension but is slow
            #     H[field][ind] = val

    # Prepend H with H0 
    H['sim_id'][:len(H0)] = np.arange(0,len(H0))
    H['given'][:len(H0)] = 1
    H['returned'][:len(H0)] = 1

    H['sim_id'][-L:] = -1
    H['given_time'][-L:] = np.inf

    H_ind = len(H0)
    start_time = time.time()
    term_test = lambda H, H_ind: termination_test(H, H_ind, exit_criteria, start_time, len(H0))

    idle_w = alloc_specs['worker_ranks'].copy()
    active_w = {EVAL_GEN_TAG:set(), EVAL_SIM_TAG:set(), 'blocked':set()}

    return H, H_ind, term_test, idle_w, active_w

def final_receive_and_kill(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test, alloc_specs):
    """ 
    Tries to receive from any active workers. 

    If time expires before all active workers have been received from, a
    nonblocking receive is posted (though the manager will not receive this
    data) and a kill signal is sent. 
    """

    exit_flag = 0

    ### Receive from all active workers 
    while len(active_w[EVAL_SIM_TAG] | active_w[EVAL_GEN_TAG]):
        H, H_ind, active_w, idle_w = receive_from_sim_and_gen(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs)
        if term_test(H, H_ind) == 2 and len(active_w[EVAL_SIM_TAG] | active_w[EVAL_GEN_TAG]):
            for w in active_w[EVAL_SIM_TAG] | active_w[EVAL_GEN_TAG]:
                comm.irecv(source=w, tag=MPI.ANY_TAG)

            print("Termination due to elapsed_wallclock_time has occurred.\n"\
              "A last attempt has been made to receive any completed work.\n"\
              "Posting nonblocking receives and kill messages for all active workers\n")
            exit_flag = 2
            break

    ### Stop all workers 
    for w in alloc_specs['worker_ranks']:
        comm.send(obj=None, dest=w, tag=STOP_TAG)

    return H[:H_ind], exit_flag
