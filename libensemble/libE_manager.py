"""
libEnsemble manager routines
====================================================
"""

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI
import numpy as np
import time, sys, os
import copy

# from message_numbers import EVAL_TAG # manager tells worker to evaluate the point 
from libensemble.message_numbers import EVAL_SIM_TAG, FINISHED_PERSISTENT_SIM_TAG
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG
from libensemble.message_numbers import PERSIS_STOP
from libensemble.message_numbers import STOP_TAG # tag for manager interupt messages to workers (sh: maybe change name)
from libensemble.message_numbers import UNSET_TAG
from libensemble.message_numbers import WORKER_KILL
from libensemble.message_numbers import WORKER_KILL_ON_ERR
from libensemble.message_numbers import WORKER_KILL_ON_TIMEOUT
from libensemble.message_numbers import JOB_FAILED 
from libensemble.message_numbers import WORKER_DONE
from libensemble.message_numbers import MAN_SIGNAL_FINISH # manager tells worker run is over
from libensemble.message_numbers import MAN_SIGNAL_KILL # manager tells worker to kill running job/jobs
from libensemble.calc_info import CalcInfo

def manager_main(libE_specs, alloc_specs, sim_specs, gen_specs, failure_processing, exit_criteria, H0):
    """
    Manager routine to coordinate the generation and simulation evaluations
    """

    #quick - until do proper timer
    man_start_time = time.time()
    
    H, H_ind, term_test, worker_sets, comm = initialize(sim_specs, gen_specs, alloc_specs, exit_criteria, H0, libE_specs)
    persistent_queue_data = {}; gen_info = {}

    send_initial_info_to_workers(comm, H, sim_specs, gen_specs, worker_sets)

    ### Continue receiving and giving until termination test is satisfied
    while not term_test(H, H_ind):

        H, H_ind, worker_sets, gen_info = receive_from_sim_and_gen(comm, worker_sets, H, H_ind, sim_specs, gen_specs, gen_info)

        persistent_queue_data = update_active_and_queue(H[:H_ind], libE_specs, gen_specs, persistent_queue_data)

        Work, gen_info = alloc_specs['alloc_f'](worker_sets, H[:H_ind], sim_specs, gen_specs, gen_info)

        for w in Work:
            if term_test(H,H_ind):
                break
            worker_sets = send_to_worker_and_update_active_and_idle(comm, H, Work[w], w, sim_specs, gen_specs, worker_sets)

    H, gen_info, exit_flag = final_receive_and_kill(comm, worker_sets, H, H_ind, sim_specs, gen_specs, term_test, libE_specs, gen_info, man_start_time)

    return H, gen_info, exit_flag




######################################################################
# Manager subroutines
######################################################################

def send_initial_info_to_workers(comm, H, sim_specs, gen_specs, worker_sets):
    for w in worker_sets['nonpersis_w']['waiting']:
        comm.send(obj=H[sim_specs['in']].dtype, dest=w)
        comm.send(obj=H[gen_specs['in']].dtype, dest=w)


def send_to_worker_and_update_active_and_idle(comm, H, Work, w, sim_specs, gen_specs, worker_sets):
    """
    Sends calculation information to the workers and updates the sets of
    active/idle workers
    """
    
    work_rows = Work['libE_info']['H_rows']
    
    if len(work_rows):            
        assert set(Work['H_fields']).issubset(H.dtype.names), "Allocation function requested the field(s): " + str(list(set(Work['H_fields']).difference(H.dtype.names))) + " be sent to worker=" + str(w) + ", but this field is not in history"
        calc_in = H[Work['H_fields']][work_rows]
    else:
        calc_in = None
        
    comm.send(obj=Work, dest=w, tag=Work['tag']) #Kept tag for now but NOT going to use it like this
    if len(work_rows):
        comm.send(obj=H[Work['H_fields']][work_rows],dest=w)

    # Remove worker from either 'waiting' set and add it to the appropriate 'active' set
    worker_sets['nonpersis_w']['waiting'].difference_update([w]); 
    worker_sets['persis_w']['waiting'][Work['tag']].difference_update([w])
    if 'libE_info' in Work and 'persistent' in Work['libE_info']:
        worker_sets['persis_w'][Work['tag']].add(w)
    else:
        worker_sets['nonpersis_w'][Work['tag']].add(w)

    if 'blocking' in Work['libE_info']:
        worker_sets['nonpersis_w']['blocked'].update(Work['libE_info']['blocking'])
        worker_sets['nonpersis_w']['waiting'].difference_update(Work['libE_info']['blocking'])

    if Work['tag'] == EVAL_SIM_TAG:
        update_history_x_out(H, work_rows, w)

    return worker_sets 


def receive_from_sim_and_gen(comm, worker_sets, H, H_ind, sim_specs, gen_specs, gen_info):
    """
    Receive calculation output from workers. Loops over all active workers and
    probes to see if worker is ready to communticate. If any output is
    received, all other workers are looped back over.
    """
    status = MPI.Status()

    new_stuff = True
    while new_stuff and len(worker_sets['nonpersis_w'][EVAL_SIM_TAG] | worker_sets['nonpersis_w'][EVAL_GEN_TAG] | worker_sets['persis_w'][EVAL_SIM_TAG] | worker_sets['persis_w'][EVAL_GEN_TAG]) > 0:
        new_stuff = False
        for w in worker_sets['nonpersis_w'][EVAL_SIM_TAG] | worker_sets['nonpersis_w'][EVAL_GEN_TAG] | worker_sets['persis_w'][EVAL_SIM_TAG] | worker_sets['persis_w'][EVAL_GEN_TAG]: 
            if comm.Iprobe(source=w, tag=MPI.ANY_TAG, status=status):
                new_stuff = True

                D_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
                #print('D_recv',D_recv)
                calc_type = D_recv['calc_type']
                calc_status = D_recv['calc_status']
                #recv_tag = status.Get_tag()
                
                assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], 'Aborting, Unknown calculation type received. Received type: ' + str(calc_type)
                
                assert calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG, UNSET_TAG, MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, WORKER_KILL, JOB_FAILED, WORKER_DONE], 'Aborting: Unknown calculation status received. Received status: ' + str(calc_status)
                
                #assert recv_tag in [EVAL_SIM_TAG, EVAL_GEN_TAG, FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG], 'Unknown calculation tag received. Exiting'
                
                if calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG]:
                    worker_sets['persis_w'][EVAL_GEN_TAG].difference_update([w])
                    worker_sets['persis_w'][EVAL_SIM_TAG].difference_update([w])
                    worker_sets['nonpersis_w']['waiting'].add(w)
                else:
                    
                    if calc_type in [EVAL_SIM_TAG]:
                        update_history_f(H, D_recv)
                    
                    if calc_type in [EVAL_GEN_TAG]:
                        H, H_ind = update_history_x_in(H, H_ind, w, D_recv['calc_out']) 
                        
                    if 'libE_info' in D_recv and 'persistent' in D_recv['libE_info']:
                        worker_sets['persis_w']['waiting'][calc_type].add(w)
                        worker_sets['persis_w'][calc_type].remove(w)
                    else:
                        worker_sets['nonpersis_w']['waiting'].add(w)
                        worker_sets['nonpersis_w'][calc_type].remove(w)                        

                if 'libE_info' in D_recv and 'blocking' in D_recv['libE_info']:
                        worker_sets['nonpersis_w']['blocked'].difference_update(D_recv['libE_info']['blocking'])
                        worker_sets['nonpersis_w']['waiting'].update(D_recv['libE_info']['blocking'])

                if 'gen_info' in D_recv:
                    for key in D_recv['gen_info'].keys():
                        gen_info[w][key] = D_recv['gen_info'][key]


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

    return H, H_ind, worker_sets, gen_info


def update_active_and_queue(H, libE_specs, gen_specs, data):
    """ 
    Call a user-defined function that decides if active work should be continued
    and possibly updated the priority of points in H.
    """
    if 'queue_update_function' in libE_specs and len(H):
        H, data = libE_specs['queue_update_function'](H,gen_specs, data)
    
    return data


def update_history_f(H, D): 
    """
    Updates the history (in place) after a point has been evaluated
    """

    new_inds = D['libE_info']['H_rows']
    H_0 = D['calc_out']

    for j,ind in enumerate(new_inds): 
        for field in H_0.dtype.names:
            H[field][ind] = H_0[field][j]

        H['returned'][ind] = True


def update_history_x_out(H, q_inds, sim_worker):
    """
    Updates the history (in place) when a new point has been given out to be evaluated

    """

    for i,j in zip(q_inds,range(len(q_inds))):
        H['given'][i] = True
        H['given_time'][i] = time.time()
        H['sim_worker'][i] = sim_worker


def update_history_x_in(H, H_ind, gen_worker, O):
    """
    Updates the history (in place) when a new point has been returned from a gen

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    H_ind: integer
        The new point
    gen_worker: integer
        The worker who generated these points
    O: numpy array
        Output from gen_func
    """

    if len(O) == 0:
        return H, H_ind

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

    H['gen_worker'][update_inds] = gen_worker

    H_ind += num_new

    return H, H_ind


def grow_H(H, k):
    """ 
    libEnsemble is requesting k rows be added to H because the gen_func produced
    more points than rows in H.
    """
    H_1 = np.zeros(k, dtype=H.dtype)
    H_1['sim_id'] = -1
    H_1['given_time'] = np.inf

    H = np.append(H,H_1)

    return H


def termination_test(H, H_ind, exit_criteria, start_time, lenH0):
    """
    Return nonzero if the libEnsemble run should stop 
    """

    if 'elapsed_wallclock_time' in exit_criteria:
        if time.time() - start_time >= exit_criteria['elapsed_wallclock_time']:
            return 2
        
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

    return False


def initialize(sim_specs, gen_specs, alloc_specs, exit_criteria, H0, libE_specs):
    """
    Forms the numpy structured array that records everything from the
    libEnsemble run 

    Returns
    ----------
    H: numpy structured array
        History array storing rows for each point. Field names are in
        libensemble/libE_fields.py

    H_ind: integer
        Where libEnsemble should start filling in H

    term_test: lambda funciton
        Simplified termination test (doesn't require passing fixed quantities).
        This is nice when calling term_test in multiple places.

    idle_w: python set
        Idle worker (initially all workers)

    active_w: python set
        Active worker (initially empty)
    """

    if 'sim_max' in exit_criteria:
        L = exit_criteria['sim_max']
    else:
        L = 100

    from libensemble.libE_fields import libE_fields

    H = np.zeros(L + len(H0), dtype=list(set(libE_fields + sim_specs['out'] + gen_specs['out'] + alloc_specs['out'])))

    if len(H0):
        fields = H0.dtype.names

        for field in fields:
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

    worker_sets = {}
    worker_sets['nonpersis_w'] = {'waiting': libE_specs['worker_ranks'].copy(), EVAL_GEN_TAG:set(), EVAL_SIM_TAG:set(), 'blocked':set()}
    worker_sets['persis_w'] = {'waiting':{EVAL_SIM_TAG:set(),EVAL_GEN_TAG:set()}, EVAL_SIM_TAG:set(), EVAL_GEN_TAG:set()}

    comm = libE_specs['comm']

    return H, H_ind, term_test, worker_sets, comm


##Create a utils module for stuff like this
#def smart_sort(l):
    #import re
    #""" Sort the given iterable in the way that humans expect.
    
    #For example: Worker10 comes after Worker9. No padding required
    #""" 
    #convert = lambda text: int(text) if text.isdigit() else text 
    #alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    #return sorted(l, key = alphanum_key)


def final_receive_and_kill(comm, worker_sets, H, H_ind, sim_specs, gen_specs, term_test, libE_specs, gen_info, man_start_time):
    """ 
    Tries to receive from any active workers. 

    If time expires before all active workers have been received from, a
    nonblocking receive is posted (though the manager will not receive this
    data) and a kill signal is sent. 
    """

    exit_flag = 0

    ### Receive from all active workers 
    while len(worker_sets['nonpersis_w'][EVAL_SIM_TAG] | worker_sets['nonpersis_w'][EVAL_GEN_TAG] | worker_sets['persis_w'][EVAL_SIM_TAG] | worker_sets['persis_w'][EVAL_GEN_TAG]):
        H, H_ind, worker_sets, gen_info = receive_from_sim_and_gen(comm, worker_sets, H, H_ind, sim_specs, gen_specs, gen_info)
        if term_test(H, H_ind) == 2 and len(worker_sets['nonpersis_w'][EVAL_SIM_TAG] | worker_sets['nonpersis_w'][EVAL_GEN_TAG] | worker_sets['persis_w'][EVAL_SIM_TAG] | worker_sets['persis_w'][EVAL_GEN_TAG]):
            print("Termination due to elapsed_wallclock_time has occurred.\n"\
              "A last attempt has been made to receive any completed work.\n"\
              "Posting nonblocking receives and kill messages for all active workers\n")
            sys.stdout.flush()
            sys.stderr.flush()            

            for w in worker_sets['nonpersis_w'][EVAL_SIM_TAG] | worker_sets['nonpersis_w'][EVAL_GEN_TAG] | worker_sets['persis_w'][EVAL_SIM_TAG] | worker_sets['persis_w'][EVAL_GEN_TAG]:
                comm.irecv(source=w, tag=MPI.ANY_TAG)
            exit_flag = 2
            break
    
    ### Kill the workers
    for w in libE_specs['worker_ranks']:
        stop_signal = MAN_SIGNAL_FINISH
        comm.send(obj=stop_signal, dest=w, tag=STOP_TAG)
       
    print("\nlibEnsemble manager total time:", time.time() - man_start_time)
       
    # Create calc summary file
    time.sleep(5)
    CalcInfo.merge_statfiles()

    return H[:H_ind], gen_info, exit_flag
