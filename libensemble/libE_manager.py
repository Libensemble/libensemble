"""
libEnsemble manager routines
====================================================
"""

from __future__ import division
from __future__ import absolute_import

import time, sys, os
import logging
import socket
import pickle

from mpi4py import MPI
import numpy as np

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

logger = logging.getLogger(__name__)
#For debug messages - uncomment
# logger.setLevel(logging.DEBUG)

def manager_main(hist, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, persis_info):
    """
    Manager routine to coordinate the generation and simulation evaluations
    """

    man_start_time = time.time()
    
    term_test, W, comm = initialize(hist, sim_specs, gen_specs, alloc_specs, exit_criteria, libE_specs)
    
    logger.info("Manager initiated on MPI rank {} on node {}".format(comm.Get_rank(), socket.gethostname()))
    logger.info("Manager exit_criteria: {}".format(exit_criteria))

    persistent_queue_data = {}

    send_initial_info_to_workers(comm, hist, sim_specs, gen_specs)

    ### Continue receiving and giving until termination test is satisfied
    while not term_test(hist):

        W, persis_info = receive_from_sim_and_gen(comm, W, hist, sim_specs, gen_specs, persis_info)

        persistent_queue_data = update_active_and_queue(hist.trim_H(), libE_specs, gen_specs, persistent_queue_data)        

        if any(W['active'] == 0):
            Work, persis_info = alloc_specs['alloc_f'](W, hist.trim_H(), sim_specs, gen_specs, persis_info)

            for w in Work:
                if term_test(hist):
                    break
                W = send_to_worker_and_update_active_and_idle(comm, hist, Work[w], w, sim_specs, gen_specs, W)

    persis_info, exit_flag = final_receive_and_kill(comm, W, hist, sim_specs, gen_specs, term_test, libE_specs, persis_info, man_start_time)

    return persis_info, exit_flag




######################################################################
# Manager subroutines
######################################################################


def send_initial_info_to_workers(comm, hist, sim_specs, gen_specs):
    comm.bcast(obj=hist.H[sim_specs['in']].dtype)
    comm.bcast(obj=hist.H[gen_specs['in']].dtype)


def send_to_worker_and_update_active_and_idle(comm, hist, Work, w, sim_specs, gen_specs, W):
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
        assert set(Work['H_fields']).issubset(hist.H.dtype.names), "Allocation function requested the field(s): " + str(list(set(Work['H_fields']).difference(hist.H.dtype.names))) + " be sent to worker=" + str(w) + ", but this field is not in history"
        comm.send(obj=hist.H[Work['H_fields']][work_rows], dest=w)

    W[w-1]['active'] = Work['tag']

    if 'libE_info' in Work and 'persistent' in Work['libE_info']:
        W[w-1]['persis_state'] = Work['tag']

    if 'blocking' in Work['libE_info']:
        for w_i in Work['libE_info']['blocking']:
            assert W[w_i-1]['active'] == 0, "Active worker being blocked; aborting"
            W[w_i-1]['blocked'] = 1
            W[w_i-1]['active'] = 1

    if Work['tag'] == EVAL_SIM_TAG:
        hist.update_history_x_out(work_rows, w)

    return W


def receive_from_sim_and_gen(comm, W, hist, sim_specs, gen_specs, persis_info):

    """
    Receive calculation output from workers. Loops over all active workers and
    probes to see if worker is ready to communticate. If any output is
    received, all other workers are looped back over.
    """
    status = MPI.Status()

    new_stuff = True
    while new_stuff and any(W['active']):
        new_stuff = False
        for w in W['worker_id'][W['active'] > 0]:
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

                calc_type = D_recv['calc_type']
                calc_status = D_recv['calc_status']

                assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], 'Aborting, Unknown calculation type received. Received type: ' + str(calc_type)

                assert calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG, UNSET_TAG, MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, WORKER_KILL, JOB_FAILED, WORKER_DONE], 'Aborting: Unknown calculation status received. Received status: ' + str(calc_status)

                W[w-1]['active'] = 0
                if calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG]:
                    W[w-1]['persis_state'] = 0
                else:

                    if calc_type in [EVAL_SIM_TAG]:
                        hist.update_history_f(D_recv)

                    if calc_type in [EVAL_GEN_TAG]:
                        hist.update_history_x_in(w, D_recv['calc_out'])

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
        #count = k*(sum(hist.H['returned'])//k)
        count = k*(hist.sim_count//k)        
        filename = 'libE_history_after_sim_' + str(count) + '.npy'

        if not os.path.isfile(filename) and count > 0:
            np.save(filename, hist.H)

    if 'save_every_k' in gen_specs:
        k = gen_specs['save_every_k']
        count = k*(hist.index//k)
        filename = 'libE_history_after_gen_' + str(count) + '.npy'

        if not os.path.isfile(filename) and count > 0:
            np.save(filename, hist.H)
            
    return W, persis_info


def update_active_and_queue(H, libE_specs, gen_specs, data):
    """
    Call a user-defined function that decides if active work should be continued
    and possibly updated the priority of points in H.
    """
    if 'queue_update_function' in libE_specs and len(H):
        data = libE_specs['queue_update_function'](H, gen_specs, data)

    return data


def termination_test(hist, exit_criteria, start_time):
    """
    Return nonzero if the libEnsemble run should stop
    """

    # Time should be checked first to ensure proper timeout
    if 'elapsed_wallclock_time' in exit_criteria:
        if time.time() - start_time >= exit_criteria['elapsed_wallclock_time']:
            logger.debug("Term test tripped: elapsed_wallclock_time")
            return 2

    if 'sim_max' in exit_criteria:
        if hist.given_count >= exit_criteria['sim_max'] + hist.offset:
            logger.debug("Term test tripped: sim_max")
            return 1

    if 'gen_max' in exit_criteria:
        if hist.index >= exit_criteria['gen_max'] + hist.offset:
            logger.debug("Term test tripped: gen_max")
            return 1

    if 'stop_val' in exit_criteria:
        key = exit_criteria['stop_val'][0]
        val = exit_criteria['stop_val'][1]
        if np.any(hist.H[key][:hist.index][~np.isnan(hist.H[key][:hist.index])] <= val):
            logger.debug("Term test tripped: stop_val")
            return 1

    return False


# Can remove more args if dont add hist setup option in here: Not using: sim_specs, gen_specs, alloc_specs
def initialize(hist, sim_specs, gen_specs, alloc_specs, exit_criteria, libE_specs):
    """
    Forms the numpy structured array that records everything from the
    libEnsemble run

    Returns
    ----------
    hist: History object
        LibEnsembles History data structure
        
    term_test: lambda funciton
        Simplified termination test (doesn't require passing fixed quantities).
        This is nice when calling term_test in multiple places.

    worker_sets: python set
        Data structure containing lists of active and idle workers
        Initially all workers are idle
    
    comm: MPI communicator
        The communicator for libEnsemble manager and workers
    """

    start_time = time.time()
    term_test = lambda hist: termination_test(hist, exit_criteria, start_time)

    W = np.zeros(len(libE_specs['workers']),
                 dtype=[('worker_id', int), ('active', int), ('persis_state', int), ('blocked', bool)])
    W['worker_id'] = sorted(libE_specs['workers'])

    comm = libE_specs['comm']

    return term_test, W, comm


def final_receive_and_kill(comm, W, hist, sim_specs, gen_specs, term_test, libE_specs, persis_info, man_start_time):
    """
    Tries to receive from any active workers.

    If time expires before all active workers have been received from, a
    nonblocking receive is posted (though the manager will not receive this
    data) and a kill signal is sent.
    """

    exit_flag = 0

    ### Receive from all active workers
    while any(W['active']):

        W, persis_info = receive_from_sim_and_gen(comm, W, hist, sim_specs, gen_specs, persis_info)

        if term_test(hist) == 2 and any(W['active']):

            print("Termination due to elapsed_wallclock_time has occurred.\n"\
              "A last attempt has been made to receive any completed work.\n"\
              "Posting nonblocking receives and kill messages for all active workers\n")
            sys.stdout.flush()
            sys.stderr.flush()

            status = MPI.Status()
            for w in W['worker_id'][W['active'] > 0]:
                if comm.Iprobe(source=w, tag=MPI.ANY_TAG, status=status):
                    D_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
            exit_flag = 2
            break

    ### Kill the workers
    for w in libE_specs['workers']:
        stop_signal = MAN_SIGNAL_FINISH
        comm.send(obj=stop_signal, dest=w, tag=STOP_TAG)

    print("\nlibEnsemble manager total time:", time.time() - man_start_time)
    return persis_info, exit_flag
