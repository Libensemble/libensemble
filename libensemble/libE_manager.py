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
import logging
import socket
import pickle

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
from libensemble.message_numbers import MAN_SIGNAL_REQ_RESEND, MAN_SIGNAL_REQ_PICKLE_DUMP
from libensemble.calc_info import CalcInfo

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#For debug messages - uncomment
# logger.setLevel(logging.DEBUG)

class ManagerException(Exception): pass

def manager_main(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0, persis_info):
    """
    Manager routine to coordinate the generation and simulation evaluations
    """

    man_start_time = time.time()
    
    H, H_ind, term_test, W, comm, given_count = initialize(sim_specs, gen_specs, alloc_specs, exit_criteria, H0, libE_specs)
    
    logger.info("Manager initiated on MPI rank {} on node {}".format(comm.Get_rank(), socket.gethostname()))
    logger.info("Manager exit_criteria: {}".format(exit_criteria))    
    
    persistent_queue_data = {}
    
    send_initial_info_to_workers(comm, H, sim_specs, gen_specs, W)

    ### Continue receiving and giving until termination test is satisfied
    while not term_test(H, H_ind, given_count):

        H, H_ind, W, persis_info = receive_from_sim_and_gen(comm, W, H, H_ind, sim_specs, gen_specs, persis_info)

        persistent_queue_data = update_active_and_queue(H[:H_ind], libE_specs, gen_specs, persistent_queue_data)

        if any(W['active']==0):
            Work, persis_info = alloc_specs['alloc_f'](W, H[:H_ind], sim_specs, gen_specs, persis_info)

            for w in Work:
                if term_test(H, H_ind, given_count):
                    break
                W, given_count = send_to_worker_and_update_active_and_idle(comm, H, Work[w], w, sim_specs, gen_specs, W, given_count)

    H, persis_info, exit_flag = final_receive_and_kill(comm, W, H, H_ind, sim_specs, gen_specs, term_test, libE_specs, persis_info, given_count, man_start_time)

    return H, persis_info, exit_flag




######################################################################
# Manager subroutines
######################################################################

def send_initial_info_to_workers(comm, H, sim_specs, gen_specs, W):
    comm.bcast(obj=H[sim_specs['in']].dtype)
    comm.bcast(obj=H[gen_specs['in']].dtype)


def send_to_worker_and_update_active_and_idle(comm, H, Work, w, sim_specs, gen_specs, W, given_count):
    """
    Sends calculation information to the workers and updates the sets of
    active/idle workers

    Note that W is indexed from 0, but the worker_ids are indexed from 1, hence the
    use of the w-1 when refering to rows in W.
    """
    assert w != 0, "Can't send to worker 0; this is the manager. Aborting"
    assert W[w-1]['active'] == 0, "Allocation function requested work to an already active worker. Aborting"
    
    logger.debug("Manager sending work unit to worker {}".format(w)) #rank
    comm.send(obj=Work, dest=w, tag=Work['tag']) 
    work_rows = Work['libE_info']['H_rows']    
    if len(work_rows):            
        assert set(Work['H_fields']).issubset(H.dtype.names), "Allocation function requested the field(s): " + str(list(set(Work['H_fields']).difference(H.dtype.names))) + " be sent to worker=" + str(w) + ", but this field is not in history"
        comm.send(obj=H[Work['H_fields']][work_rows],dest=w)
    #     for i in Work['H_fields']:
    #         # comm.send(obj=H[i][0].dtype,dest=w)
    #         comm.Send(H[i][Work['libE_info']['H_rows']], dest=w)  
    
    W[w-1]['active'] = Work['tag']

    if 'libE_info' in Work and 'persistent' in Work['libE_info']:
        W[w-1]['persis_state'] = Work['tag']

    if 'blocking' in Work['libE_info']:
        for w_i in Work['libE_info']['blocking']:
            assert W[w_i-1]['active'] == 0, "Active worker being blocked; aborting"
            W[w_i-1]['blocked'] = 1
            W[w_i-1]['active'] = 1

    if Work['tag'] == EVAL_SIM_TAG:
        update_history_x_out(H, work_rows, w)
        given_count += 1
        
    return W, given_count


def receive_from_sim_and_gen(comm, W, H, H_ind, sim_specs, gen_specs, persis_info):
    """
    Receive calculation output from workers. Loops over all active workers and
    probes to see if worker is ready to communticate. If any output is
    received, all other workers are looped back over.
    """
    status = MPI.Status()

    new_stuff = True
    while new_stuff and any(W['active']):
        new_stuff = False
        for w in W['worker_id'][W['active']>0]: 
            if comm.Iprobe(source=w, tag=MPI.ANY_TAG, status=status):
                new_stuff = True
                logger.debug("Manager receiving from Worker: {}".format(w))
                try:
                    D_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
                    logger.debug("Message size {}".format(status.Get_count()))
                except Exception as e:
                    logger.error("Exception caught on Manager receive: {}".format(e))
                    logger.error("From worker: {}".format(w)) 
                    logger.error("Message size of errored message {}".format(status.Get_count()))
                    logger.error("Message status error code {}".format(status.Get_error()))
                    
                    # Need to clear message faulty message - somehow
                    status.Set_cancelled(True) #Make sure cancelled before re-send
                    
                    # Check on working with peristent data - curently set only one to True
                    man_request_resend_on_error = False
                    man_request_pkl_dump_on_error = True
                    
                    if man_request_resend_on_error:
                        #Ideally use status.Get_source() for MPI rank - this relies on rank being workerID
                        comm.send(obj=MAN_SIGNAL_REQ_RESEND, dest=w, tag=STOP_TAG)
                        D_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
                    
                    if man_request_pkl_dump_on_error:
                        # Req worker to dump pickle file and manager reads
                        comm.send(obj=MAN_SIGNAL_REQ_PICKLE_DUMP, dest=w, tag=STOP_TAG)
                        pkl_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
                        D_recv = pickle.load(open(pkl_recv, "rb"))
                        os.remove(pkl_recv) #If want to delete file
                        
                # Manager read
                #workdir_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
                #D_recv = man_read_from_file(workdir_recv)
                    
                calc_type = D_recv['calc_type']
                calc_status = D_recv['calc_status']
                #recv_tag = status.Get_tag()
                
                assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], 'Aborting, Unknown calculation type received. Received type: ' + str(calc_type)
                
                assert calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG, UNSET_TAG, MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, WORKER_KILL, JOB_FAILED, WORKER_DONE], 'Aborting: Unknown calculation status received. Received status: ' + str(calc_status)
                
                #assert recv_tag in [EVAL_SIM_TAG, EVAL_GEN_TAG, FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG], 'Unknown calculation tag received. Exiting'
                
                W[w-1]['active'] = 0
                if calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG]:
                    W[w-1]['persis_state'] = 0
                else:
                    
                    if calc_type in [EVAL_SIM_TAG]:
                        update_history_f(H, D_recv)
                    
                    if calc_type in [EVAL_GEN_TAG]:
                        H, H_ind = update_history_x_in(H, H_ind, w, D_recv['calc_out'])
                        
                    if 'libE_info' in D_recv and 'persistent' in D_recv['libE_info']:
                        # Now a waiting, persistent worker
                        W[w-1]['persis_state'] = calc_type

                if 'libE_info' in D_recv and 'blocking' in D_recv['libE_info']:
                    # Now done blocking these workers
                    for w_i in D_recv['libE_info']['blocking']:
                        W[w_i-1]['blocked'] = 0
                        W[w_i-1]['active'] = 0

                if 'persis_info' in D_recv:
                    for key in D_recv['persis_info'].keys():
                        persis_info[w][key] = D_recv['persis_info'][key]


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

    return H, H_ind, W, persis_info


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

    new_inds = D['libE_info']['H_rows'] # The list of rows (as a numpy array)
    H_0 = D['calc_out']

    for j,ind in enumerate(new_inds): 
        for field in H_0.dtype.names:
            
            if np.isscalar(H_0[field][j]):
                H[field][ind] = H_0[field][j]
            else:
                #len or np.size
                H0_size = len(H_0[field][j])
                assert H0_size <= len(H[field][ind]), "Manager Error: Too many values received for " + field 
                assert H0_size, "Manager Error: No values in this field " + field
                if H0_size == len(H[field][ind]):
                    H[field][ind] = H_0[field][j] #ref
                else:
                    H[field][ind][:H0_size] = H_0[field][j] #Slice copy

        H['returned'][ind] = True


def update_history_x_out(H, q_inds, sim_worker):
    """
    Updates the history (in place) when a new point has been given out to be evaluated

    """

    H['given'][q_inds] = True
    H['given_time'][q_inds] = time.time()
    H['sim_worker'][q_inds] = sim_worker

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


def termination_test(H, H_ind, given_count, exit_criteria, start_time, lenH0):
    """
    Return nonzero if the libEnsemble run should stop 
    """

    # Time should be checked first to ensure proper timeout
    if 'elapsed_wallclock_time' in exit_criteria:
        if time.time() - start_time >= exit_criteria['elapsed_wallclock_time']:
            logger.debug("Term test tripped: elapsed_wallclock_time")            
            return 2

    if 'sim_max' in exit_criteria:
        if given_count >= exit_criteria['sim_max'] + lenH0:
            logger.debug("Term test tripped: sim_max")            
            return 1

    if 'gen_max' in exit_criteria:
        if H_ind >= exit_criteria['gen_max'] + lenH0:
            logger.debug("Term test tripped: gen_max")            
            return 1 

    if 'stop_val' in exit_criteria:
        key = exit_criteria['stop_val'][0]
        val = exit_criteria['stop_val'][1]
        if np.any(H[key][:H_ind][~np.isnan(H[key][:H_ind])] <= val):
            logger.debug("Term test tripped: stop_val")            
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
    given_count = len(H0)
    start_time = time.time()
    term_test = lambda H, H_ind, given_count: termination_test(H, H_ind, given_count, exit_criteria, start_time, len(H0))

    W = np.zeros(len(libE_specs['workers']), dtype=[('worker_id',int),('active',int),('persis_state',int),('blocked',bool)])
    W['worker_id'] = sorted(libE_specs['workers'])

    comm = libE_specs['comm']

    return H, H_ind, term_test, W, comm, given_count


def final_receive_and_kill(comm, W, H, H_ind, sim_specs, gen_specs, term_test, libE_specs, persis_info, given_count, man_start_time):
    """ 
    Tries to receive from any active workers. 

    If time expires before all active workers have been received from, a
    nonblocking receive is posted (though the manager will not receive this
    data) and a kill signal is sent. 
    """

    exit_flag = 0

    ### Receive from all active workers 
    while any(W['active']):
        
        H, H_ind, W, persis_info = receive_from_sim_and_gen(comm, W, H, H_ind, sim_specs, gen_specs, persis_info)
        
        if term_test(H, H_ind, given_count) == 2 and any(W['active']):
            
            print("Termination due to elapsed_wallclock_time has occurred.\n"\
              "A last attempt has been made to receive any completed work.\n"\
              "Posting nonblocking receives and kill messages for all active workers\n")
            sys.stdout.flush()
            sys.stderr.flush()            

            for w in W['worker_id'][W['active']>0]:
                comm.irecv(source=w, tag=MPI.ANY_TAG)
            exit_flag = 2
            break
    
    ### Kill the workers
    for w in libE_specs['workers']:
        stop_signal = MAN_SIGNAL_FINISH
        comm.send(obj=stop_signal, dest=w, tag=STOP_TAG)
       
    print("\nlibEnsemble manager total time:", time.time() - man_start_time)
    return H[:H_ind], persis_info, exit_flag
