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

from message_numbers import EVAL_TAG # manager tells worker to evaluate the point 
from message_numbers import STOP_TAG # manager tells worker run is over

from mpi4py import MPI
import numpy as np

import time, sys, os
import copy

def manager_main(comm, allocation_specs, sim_specs, gen_specs,
        failure_processing, exit_criteria, H0):

    H, H_ind, term_test = initialize(sim_specs, gen_specs, exit_criteria, H0)

    idle_w = allocation_specs['worker_ranks'].copy()
    active_w = {'gen':set(), 'sim':set(), 'blocked':set()}

    ### Continue receiving and giving until termination test is satisfied
    while not term_test(H, H_ind):

        H, H_ind, active_w, idle_w = receive_from_sim_and_gen(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs)

        update_active_and_queue(active_w, idle_w, H[:H_ind], gen_specs)

        Work = decide_work_and_resources(active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test)

        for w in Work:
            comm.send(obj=Work[w], dest=w, tag=EVAL_TAG)
            active_w, idle_w = update_active_and_idle(active_w, idle_w, w, Work[w])

    H, exit_flag = final_receive_and_kill(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test, allocation_specs)

    return H, exit_flag





######################################################################
# Manager subroutines
######################################################################
def receive_from_sim_and_gen(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs):
    status = MPI.Status()

    active_w_copy = copy.deepcopy(active_w)

    while True:
        for w in active_w_copy['sim'] | active_w_copy['gen']: 
            if comm.Iprobe(source=w, tag=MPI.ANY_TAG):
                D_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
                idle_w.add(status.Get_source())
                active_w[D_recv['calc_info']['type']].remove(status.Get_source())

                assert D_recv['calc_info']['type'] in ['sim','gen'], 'Unknown calculation type received. Exiting'
                if D_recv['calc_info']['type'] == 'sim':
                    update_history_f(H, D_recv)
                else: # D_recv['calc_info']['type'] == 'gen':
                    H, H_ind = update_history_x_in(H, H_ind, D_recv['calc_out'])

                if 'blocking' in D_recv['calc_info']:
                    active_w['blocked'].difference_update(D_recv['calc_info']['blocking'])
                    idle_w.update(D_recv['calc_info']['blocking'])

        if active_w_copy == active_w:
            break
        else:
            active_w_copy = copy.deepcopy(active_w)

    if 'save_every_k' in sim_specs:
        k = sim_specs['save_every_k']
        count = k*(sum(H['returned'])//k)
        filename = 'LibE_history_after_sim_' + str(count) + '.npy'

        if not os.path.isfile(filename) and count > 0:
            np.save(filename,H)

    if 'save_every_k' in gen_specs:
        k = gen_specs['save_every_k']
        count = k*(H_ind//k)
        filename = 'LibE_history_after_gen_' + str(count) + '.npy'

        if not os.path.isfile(filename) and count > 0:
            np.save(filename,H)

    return H, H_ind, active_w, idle_w


def decide_work_and_resources(active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test):
    """ Decide what should be given to workers. Note that everything put into
    the Work dictionary will be given, so we are careful not to put more gen or
    sim items into Work than necessary.
    """

    Work = {}
    gen_work = 0

    for i in idle_w:
        if term_test(H, H_ind):
            break

        blocked_set = active_w['blocked'].union(*[j['calc_info']['blocking'] for j in Work.values() if 'blocking' in j['calc_info']])
        # Only consider giving to worker i if it's resources are not blocked
        if i in blocked_set:
            continue

        q_inds = np.where(np.logical_and(~H['given'][:H_ind],~H['paused'][:H_ind]))[0]

        if len(q_inds):
            if 'priority' in H.dtype.fields:
                if 'give_all_with_same_priority' in gen_specs and gen_specs['give_all_with_same_priority']:
                    # Give all points with highest priority
                    sim_ids_to_send = q_inds[ np.where(H['priority'][q_inds] == max(H['priority'][q_inds]))[0] ]
                else:
                    # Give first point with highest priority
                    sim_ids_to_send = q_inds[ np.where(H['priority'][q_inds] == max(H['priority'][q_inds]))[0][0] ]

            else:
                # Give oldest point
                sim_ids_to_send = np.min(q_inds)
            sim_ids_to_send = np.atleast_1d(sim_ids_to_send)

            # Only give work if enough idle workers
            if 'num_nodes' in H.dtype.names and np.any(H[sim_ids_to_send]['num_nodes'] > 1):
                if np.any(H[sim_ids_to_send]['num_nodes'] > len(idle_w) - len(Work) - len(blocked_set)):
                    # Worker doesn't get gen work or anything. Just waiting for other resources to open up
                    break
                block_others = True
            else:
                block_others = False

            Work[i] = {'calc_f': sim_specs['sim_f'][0], 
                       'calc_params': sim_specs['params'], 
                       'form_subcomm': [], 
                       # 'calc_in': H[sim_specs['in']][sim_ids_to_send],
                       'calc_in': H[sim_ids_to_send][sim_specs['in']],
                       'calc_out': sim_specs['out'],
                       'calc_info': {'type':'sim', 'sim_id': sim_ids_to_send},
                      }

            if block_others:
                # import pdb; pdb.set_trace()
                unassigned_workers = idle_w - set(Work.keys()) - blocked_set
                workers_to_block = list(unassigned_workers)[:np.max(H[sim_ids_to_send]['num_nodes'])-1]
                Work[i]['calc_info']['blocking'] = set(workers_to_block)

            update_history_x_out(H, sim_ids_to_send, Work[i]['calc_in'], i, sim_specs['params'])

        else:
            # Don't give out any gen instances if in batch mode and any point has not been returned or paused
            if 'batch_mode' in gen_specs and gen_specs['batch_mode'] and any(np.logical_and(~H['returned'][:H_ind],~H['paused'][:H_ind])):
                break

            # Limit number of gen instances if given
            if 'num_inst' in gen_specs and len(active_w['gen']) + gen_work >= gen_specs['num_inst']:
                break

            # Give gen work 
            gen_work += 1 

            Work[i] = {'calc_f': gen_specs['gen_f'], 
                       'calc_params': gen_specs['params'], 
                       'form_subcomm': [], 
                       # 'calc_in': H[gen_specs['in']][:H_ind],
                       'calc_in': H[:H_ind][gen_specs['in']],
                       'calc_out': gen_specs['out'],
                       'calc_info': {'type':'gen'},
                       }


    return Work


def update_active_and_queue(active_w, idle_w, H, gen_specs):
    """ Decide if active work should be continued and the queue order

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    """
    if 'queue_update_function' in gen_specs:
        gen_specs['queue_update_function'](H,gen_specs)
    
    return


def update_history_f(H, D): 
    """
    Updates the history (in place) after a point has been evaluated

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    """

    new_inds = D['calc_info']['sim_id']
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
    """ LibEnsemble is requesting k rows be added to H because gen_f produced
    more points than rows in H."""
    H_1 = np.zeros(k, dtype=H.dtype)
    H_1['sim_id'] = -1
    H_1['given_time'] = np.inf

    H = np.append(H,H_1)

    return H


def update_history_x_out(H, q_inds, W, lead_rank, sim_f_params):
    """
    Updates the history (in place) when a new point has been given out to be evaluated

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    H_ind: integer
        The new point
    W: numpy array
        Work to be evaluated
    lead_rank: int
        lead ranks for the evaluation of x 
    """

    for i,j in zip(q_inds,range(len(W))):
        for field in W.dtype.names:
            H[field][i] = W[field][j]

        H['given'][i] = True
        H['given_time'][i] = time.time()
        H['lead_rank'][i] = lead_rank


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
        if any(H[key][:H_ind][~np.isnan(H[key][:H_ind])] <= val): 
            return 1

    if 'elapsed_wallclock_time' in exit_criteria:
        if time.time() - start_time >= exit_criteria['elapsed_wallclock_time']:
            return 2

    return False


def initialize(sim_specs, gen_specs, exit_criteria, H0):
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
        Where LibE should start filling in H

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
            assert np.max(H0['obj_component']) < gen_specs['params']['components'], "H0 has more obj_components than exist for this problem. Exiting."

        for field in fields:
            assert H[field].ndim == H0[field].ndim, "H0 and H have different ndim for field: " + field + ". Exiting"
            assert np.all(np.array(H[field].shape) >= np.array(H0[field].shape)), "H is not large enough to receive all of the components of H0 in field: " + field + ". Exiting"

            for ind, val in np.ndenumerate(H0[field]): # Works if H0[field] has arbitrary dimension
                H[field][ind] = val

    # Prepend H with H0 
    H['sim_id'][:len(H0)] = np.arange(0,len(H0))
    H['given'][:len(H0)] = 1
    H['returned'][:len(H0)] = 1

    H['sim_id'][-L:] = -1
    H['given_time'][-L:] = np.inf

    H_ind = len(H0)
    start_time = time.time()
    term_test = lambda H, H_ind: termination_test(H, H_ind, exit_criteria, start_time, len(H0))

    return (H, H_ind, term_test)

def update_active_and_idle(active_w, idle_w, w, Work):

    active_w[Work['calc_info']['type']].add(w)
    idle_w.remove(w)

    if 'blocking' in Work['calc_info']:
        active_w['blocked'].update(Work['calc_info']['blocking'])
        idle_w.difference_update(Work['calc_info']['blocking'])

    return active_w, idle_w

def final_receive_and_kill(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test, allocation_specs):
    """ 
    Tries to receive from any active workers. 

    If time expires before all active workers have been received from, a
    nonblocking receive is posted (though the manager will not receive this
    data) and a kill signal is sent. 
    """

    exit_flag = 0

    ### Receive from all active workers 
    while len(active_w['sim'] | active_w['gen']):
        H, H_ind, active_w, idle_w = receive_from_sim_and_gen(comm, active_w, idle_w, H, H_ind, sim_specs, gen_specs)
        if term_test(H, H_ind) == 2 and len(active_w['sim'] | active_w['gen']):
            for w in active_w['sim'] | active_w['gen']:
                comm.irecv(source=w, tag=MPI.ANY_TAG)

            print("Termination due to elapsed_wallclock_time has occurred.\n"\
              "A last attempt has been made to receive any completed work.\n"\
              "Posting nonblocking receives and kill messages for all active workers\n")
            exit_flag = 2
            break

    ### Stop all workers 
    for w in allocation_specs['worker_ranks']:
        comm.send(obj=None, dest=w, tag=STOP_TAG)

    return H[:H_ind], exit_flag
