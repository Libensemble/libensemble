"""
libEnsemble manager routines
====================================================
"""

from __future__ import division
from __future__ import absolute_import

import time
import sys
import os
import logging
import socket
import pickle

from mpi4py import MPI
import numpy as np

# from message_numbers import EVAL_TAG # manager tells worker to evaluate the point
from libensemble.message_numbers import EVAL_SIM_TAG, FINISHED_PERSISTENT_SIM_TAG
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG
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
    comm = libE_specs['comm']

    worker_dtype = [('worker_id', int), ('active', int), ('persis_state', int), ('blocked', bool)]
    num_workers = comm.Get_size()-1
    W = np.zeros(num_workers, dtype=worker_dtype)
    W['worker_id'] = np.arange(num_workers) + 1

    def elapsed():
        """Return time since manager start"""
        return time.time()-man_start_time

    def term_test_wallclock():
        """Check against wallclock timeout"""
        if ('elapsed_wallclock_time' in exit_criteria
                and elapsed() >= exit_criteria['elapsed_wallclock_time']):
            logger.debug("Term test tripped: elapsed_wallclock_time")
            return 2
        return 0

    def term_test_sim_max():
        """Check against max simulations"""
        if ('sim_max' in exit_criteria
                and hist.given_count >= exit_criteria['sim_max'] + hist.offset):
            logger.debug("Term test tripped: sim_max")
            return 1
        return 0

    def term_test_gen_max():
        """Check against max generator calls."""
        if ('gen_max' in exit_criteria
                and hist.index >= exit_criteria['gen_max'] + hist.offset):
            logger.debug("Term test tripped: gen_max")
            return 1
        return 0

    def term_test_stop_val():
        """Check against stop value criterion."""
        if 'stop_val' in exit_criteria:
            key = exit_criteria['stop_val'][0]
            val = exit_criteria['stop_val'][1]
            if np.any(hist.H[key][:hist.index][~np.isnan(hist.H[key][:hist.index])] <= val):
                logger.debug("Term test tripped: stop_val")
                return 1
        return 0

    def term_test():
        """Check termination criteria"""
        # Time should be checked first to ensure proper timeout
        return (term_test_wallclock()
                or term_test_sim_max()
                or term_test_gen_max()
                or term_test_stop_val())

    def kill_workers():
        """Kill the workers"""
        for w in W['worker_id']:
            stop_signal = MAN_SIGNAL_FINISH
            comm.send(obj=stop_signal, dest=w, tag=STOP_TAG)

    def read_final_messages():
        """Read final messages from any active workers"""
        status = MPI.Status()
        for w in W['worker_id'][W['active'] > 0]:
            if comm.Iprobe(source=w, tag=MPI.ANY_TAG, status=status):
                comm.recv(source=w, tag=MPI.ANY_TAG, status=status)

    def print_wallclock_term():
        """Print termination message for wall clock elapsed."""
        print("Termination due to elapsed_wallclock_time has occurred.\n"\
              "A last attempt has been made to receive any completed work.\n"\
              "Posting nonblocking receives and kill messages for all active workers\n")
        sys.stdout.flush()
        sys.stderr.flush()

    def check_work_order(Work, w):
        """Check validity of an allocation function order.
        """
        assert w != 0, "Can't send to worker 0; this is the manager. Aborting"
        assert W[w-1]['active'] == 0, "Allocation function requested work to an already active worker. Aborting"
        work_rows = Work['libE_info']['H_rows']
        if len(work_rows):
            work_fields = set(Work['H_fields'])
            hist_fields = hist.H.dtype.names
            diff_fields = list(work_fields.difference(hist_fields))
            assert not diff_fields, \
              "Allocation function requested invalid fields {} be sent to worker={}.".format(diff_fields, w)

    def send_work_order(Work, w):
        """Send an allocation function order to a worker.
        """
        logger.debug("Manager sending work unit to worker {}".format(w)) #rank
        comm.send(obj=Work, dest=w, tag=Work['tag'])
        work_rows = Work['libE_info']['H_rows']
        if len(work_rows):
            comm.send(obj=hist.H[Work['H_fields']][work_rows], dest=w)

    def update_active_and_idle(Work, w):
        """Update the active/idle status of workers following an allocation order."""

        W[w-1]['active'] = Work['tag']
        if 'libE_info' in Work and 'persistent' in Work['libE_info']:
            W[w-1]['persis_state'] = Work['tag']

        if 'blocking' in Work['libE_info']:
            for w_i in Work['libE_info']['blocking']:
                assert W[w_i-1]['active'] == 0, "Active worker being blocked; aborting"
                W[w_i-1]['blocked'] = 1
                W[w_i-1]['active'] = 1

        if Work['tag'] == EVAL_SIM_TAG:
            work_rows = Work['libE_info']['H_rows']
            hist.update_history_x_out(work_rows, w)

    def final_receive_and_kill(persis_info):
        """
        Tries to receive from any active workers.

        If time expires before all active workers have been received from, a
        nonblocking receive is posted (though the manager will not receive this
        data) and a kill signal is sent.
        """
        exit_flag = 0
        while any(W['active']) and exit_flag == 0:
            persis_info = receive_from_sim_and_gen(comm, W, hist, sim_specs, gen_specs, persis_info)
            if term_test() == 2 and any(W['active']):
                print_wallclock_term()
                read_final_messages()
                exit_flag = 2

        kill_workers()
        print("\nlibEnsemble manager total time:", time.time() - man_start_time)
        return persis_info, exit_flag

    logger.info("Manager initiated on MPI rank {} on node {}".format(comm.Get_rank(), socket.gethostname()))
    logger.info("Manager exit_criteria: {}".format(exit_criteria))
    persistent_queue_data = {}

    # Send initial info to workers
    comm.bcast(obj=hist.H[sim_specs['in']].dtype)
    comm.bcast(obj=hist.H[gen_specs['in']].dtype)

    ### Continue receiving and giving until termination test is satisfied
    while not term_test():
        persis_info = receive_from_sim_and_gen(comm, W, hist, sim_specs, gen_specs, persis_info)
        trimmed_H = hist.trim_H()
        if 'queue_update_function' in libE_specs and len(trimmed_H):
            persistent_queue_data = libE_specs['queue_update_function'](trimmed_H, gen_specs, persistent_queue_data)
        if any(W['active'] == 0):
            Work, persis_info = alloc_specs['alloc_f'](W, hist.trim_H(), sim_specs, gen_specs, persis_info)
            for w in Work:
                if term_test():
                    break
                check_work_order(Work[w], w)
                send_work_order(Work[w], w)
                update_active_and_idle(Work[w], w)

    # Return persis_info, exit_flag
    return final_receive_and_kill(persis_info)


######################################################################
# Manager subroutines
######################################################################


def save_every_k(fname, hist, count, k):
    "Save history every kth step."
    count = k*(count//k)
    filename = fname.format(count)
    if not os.path.isfile(filename) and count > 0:
        np.save(filename, hist.H)


def _man_request_resend_on_error(comm, w, status=None):
    "Request the worker resend data on error."
    #Ideally use status.Get_source() for MPI rank - this relies on rank being workerID
    status = status or MPI.Status()
    comm.send(obj=MAN_SIGNAL_REQ_RESEND, dest=w, tag=STOP_TAG)
    return comm.recv(source=w, tag=MPI.ANY_TAG, status=status)


def _man_request_pkl_dump_on_error(comm, w, status=None):
    "Request the worker dump a pickle on error."
    # Req worker to dump pickle file and manager reads
    status = status or MPI.Status()
    comm.send(obj=MAN_SIGNAL_REQ_PICKLE_DUMP, dest=w, tag=STOP_TAG)
    pkl_recv = comm.recv(source=w, tag=MPI.ANY_TAG, status=status)
    D_recv = pickle.load(open(pkl_recv, "rb"))
    os.remove(pkl_recv) #If want to delete file
    return D_recv


def check_received_calc(D_recv):
    "Check the type and status fields on a receive calculation."
    calc_type = D_recv['calc_type']
    calc_status = D_recv['calc_status']
    assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], \
      'Aborting, Unknown calculation type received. Received type: ' + str(calc_type)
    assert calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG, \
                           UNSET_TAG, MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL, \
                           WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, WORKER_KILL, \
                           JOB_FAILED, WORKER_DONE], \
      'Aborting: Unknown calculation status received. Received status: ' + str(calc_status)


def _handle_msg_from_worker(comm, hist, persis_info, w, W, status):
    """Handle a message from worker w.
    """
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

        # Check on working with peristent data - curently only use one
        #D_recv = _man_request_resend_on_error(comm, w, status)
        D_recv = _man_request_pkl_dump_on_error(comm, w, status)

    calc_type = D_recv['calc_type']
    calc_status = D_recv['calc_status']
    check_received_calc(D_recv)

    W[w-1]['active'] = 0
    if calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG]:
        W[w-1]['persis_state'] = 0
    else:
        if calc_type == EVAL_SIM_TAG:
            hist.update_history_f(D_recv)
        if calc_type == EVAL_GEN_TAG:
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
                _handle_msg_from_worker(comm, hist, persis_info, w, W, status)

    if 'save_every_k' in sim_specs:
        save_every_k('libE_history_after_sim_{}.npy', hist, hist.sim_count, sim_specs['save_every_k'])
    if 'save_every_k' in gen_specs:
        save_every_k('libE_history_after_gen_{}.npy', hist, hist.index, gen_specs['save_every_k'])

    return persis_info


# DSB -- done
def termination_test(hist, exit_criteria, start_time):
    """
    Return nonzero if the libEnsemble run should stop
    """

    # Time should be checked first to ensure proper timeout
    if ('elapsed_wallclock_time' in exit_criteria
            and time.time() - start_time >= exit_criteria['elapsed_wallclock_time']):
        logger.debug("Term test tripped: elapsed_wallclock_time")
        return 2

    if ('sim_max' in exit_criteria
            and hist.given_count >= exit_criteria['sim_max'] + hist.offset):
        logger.debug("Term test tripped: sim_max")
        return 1

    if ('gen_max' in exit_criteria
            and hist.index >= exit_criteria['gen_max'] + hist.offset):
        logger.debug("Term test tripped: gen_max")
        return 1

    if 'stop_val' in exit_criteria:
        key = exit_criteria['stop_val'][0]
        val = exit_criteria['stop_val'][1]
        if np.any(hist.H[key][:hist.index][~np.isnan(hist.H[key][:hist.index])] <= val):
            logger.debug("Term test tripped: stop_val")
            return 1

    return False


# DSB -- done
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
    worker_dtype = [('worker_id', int), ('active', int), ('persis_state', int), ('blocked', bool)]
    start_time = time.time()
    term_test = lambda hist: termination_test(hist, exit_criteria, start_time)
    num_workers = libE_specs['comm'].Get_size()-1
    W = np.zeros(num_workers, dtype=worker_dtype)
    W['worker_id'] = np.arange(num_workers) + 1
    comm = libE_specs['comm']
    return term_test, W, comm
