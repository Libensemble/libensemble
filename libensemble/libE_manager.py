"""
libEnsemble manager routines
====================================================
"""

import sys
import os
import logging
import socket
import numpy as np

from libensemble.util.timer import Timer
from libensemble.message_numbers import \
    EVAL_SIM_TAG, FINISHED_PERSISTENT_SIM_TAG, \
    EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, \
    STOP_TAG, UNSET_TAG, \
    WORKER_KILL, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, \
    JOB_FAILED, WORKER_DONE, \
    MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL
from libensemble.comms.comms import CommFinishedException
from libensemble.libE_worker import WorkerErrMsg

logger = logging.getLogger(__name__)
# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)


class ManagerException(Exception):
    "Exception at manager, raised on abort signal from worker"


def manager_main(hist, libE_specs, alloc_specs,
                 sim_specs, gen_specs, exit_criteria, persis_info, wcomms=[]):
    """Manager routine to coordinate the generation and simulation evaluations
    """

    if 'in' not in gen_specs:
        gen_specs['in'] = []

    # Send dtypes to workers
    dtypes = {EVAL_SIM_TAG: hist.H[sim_specs['in']].dtype,
              EVAL_GEN_TAG: hist.H[gen_specs['in']].dtype}
    for wcomm in wcomms:
        wcomm.send(0, dtypes)

    # Set up and run manager
    mgr = Manager(hist, libE_specs, alloc_specs,
                  sim_specs, gen_specs, exit_criteria, wcomms)
    return mgr.run(persis_info)


def filter_nans(array):
    "Filter out NaNs from a numpy array."
    return array[~np.isnan(array)]


_WALLCLOCK_MSG = """
Termination due to elapsed_wallclock_time has occurred.
A last attempt has been made to receive any completed work.
Posting nonblocking receives and kill messages for all active workers.
"""


class Manager:
    """Manager class for libensemble."""

    worker_dtype = [('worker_id', int),
                    ('active', int),
                    ('persis_state', int),
                    ('blocked', bool)]

    def __init__(self, hist, libE_specs, alloc_specs,
                 sim_specs, gen_specs, exit_criteria,
                 wcomms=[]):
        """Initialize the manager."""
        timer = Timer()
        timer.start()
        self.hist = hist
        self.libE_specs = libE_specs
        self.alloc_specs = alloc_specs
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.exit_criteria = exit_criteria
        self.elapsed = lambda: timer.elapsed
        self.wcomms = wcomms
        self.W = np.zeros(len(self.wcomms), dtype=Manager.worker_dtype)
        self.W['worker_id'] = np.arange(len(self.wcomms)) + 1
        self.term_tests = \
            [(2, 'elapsed_wallclock_time', self.term_test_wallclock),
             (1, 'sim_max', self.term_test_sim_max),
             (1, 'gen_max', self.term_test_gen_max),
             (1, 'stop_val', self.term_test_stop_val)]

    # --- Termination logic routines

    def term_test_wallclock(self, max_elapsed):
        """Check against wallclock timeout"""
        return self.elapsed() >= max_elapsed

    def term_test_sim_max(self, sim_max):
        """Check against max simulations"""
        return self.hist.given_count >= sim_max + self.hist.offset

    def term_test_gen_max(self, gen_max):
        """Check against max generator calls."""
        return self.hist.index >= gen_max + self.hist.offset

    def term_test_stop_val(self, stop_val):
        """Check against stop value criterion."""
        key, val = stop_val
        H = self.hist.H
        idx = self.hist.index
        return np.any(filter_nans(H[key][:idx]) <= val)

    def term_test(self, logged=True):
        """Check termination criteria"""
        for retval, key, testf in self.term_tests:
            if key in self.exit_criteria:
                if testf(self.exit_criteria[key]):
                    if logged:
                        logger.info("Term test tripped: {}".format(key))
                    return retval
        return 0

    # --- Low-level communication routines

    def _kill_workers(self):
        """Kill the workers"""
        for w in self.W['worker_id']:
            self.wcomms[w-1].send(STOP_TAG, MAN_SIGNAL_FINISH)

    # --- Checkpointing logic

    def _save_every_k(self, fname, count, k):
        "Save history every kth step."
        count = k*(count//k)
        filename = fname.format(count)
        if not os.path.isfile(filename) and count > 0:
            np.save(filename, self.hist.H)

    def _save_every_k_sims(self):
        "Save history every kth sim step."
        self._save_every_k('libE_history_after_sim_{}.npy',
                           self.hist.sim_count,
                           self.sim_specs['save_every_k'])

    def _save_every_k_gens(self):
        "Save history every kth gen step."
        self._save_every_k('libE_history_after_gen_{}.npy',
                           self.hist.index,
                           self.gen_specs['save_every_k'])

    # --- Handle outgoing messages to workers (work orders from alloc)

    def _check_work_order(self, Work, w):
        """Check validity of an allocation function order.
        """
        assert w != 0, "Can't send to worker 0; this is the manager."
        assert self.W[w-1]['active'] == 0, \
            "Allocation function requested work to an already active worker."
        work_rows = Work['libE_info']['H_rows']
        if len(work_rows):
            work_fields = set(Work['H_fields'])
            hist_fields = self.hist.H.dtype.names
            diff_fields = list(work_fields.difference(hist_fields))
            assert not diff_fields, \
                "Allocation function requested invalid fields {}" \
                "be sent to worker={}.".format(diff_fields, w)

    def _send_work_order(self, Work, w):
        """Send an allocation function order to a worker.
        """
        logger.debug("Manager sending work unit to worker {}".format(w))
        self.wcomms[w-1].send(Work['tag'], Work)
        work_rows = Work['libE_info']['H_rows']
        if len(work_rows):
            self.wcomms[w-1].send(0, self.hist.H[Work['H_fields']][work_rows])

    def _update_state_on_alloc(self, Work, w):
        """Update worker active/idle status following an allocation order."""

        self.W[w-1]['active'] = Work['tag']
        if 'libE_info' in Work and 'persistent' in Work['libE_info']:
            self.W[w-1]['persis_state'] = Work['tag']

        if 'blocking' in Work['libE_info']:
            for w_i in Work['libE_info']['blocking']:
                assert self.W[w_i-1]['active'] == 0, \
                    "Active worker being blocked; aborting"
                self.W[w_i-1]['blocked'] = 1
                self.W[w_i-1]['active'] = 1

        if Work['tag'] == EVAL_SIM_TAG:
            work_rows = Work['libE_info']['H_rows']
            self.hist.update_history_x_out(work_rows, w)

    # --- Handle incoming messages from workers

    @staticmethod
    def _check_received_calc(D_recv):
        "Check the type and status fields on a receive calculation."
        calc_type = D_recv['calc_type']
        calc_status = D_recv['calc_status']
        assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], \
            "Aborting, Unknown calculation type received. " \
            "Received type: {}".format(calc_type)
        assert calc_status in [FINISHED_PERSISTENT_SIM_TAG,
                               FINISHED_PERSISTENT_GEN_TAG,
                               UNSET_TAG,
                               MAN_SIGNAL_FINISH,
                               MAN_SIGNAL_KILL,
                               WORKER_KILL_ON_ERR,
                               WORKER_KILL_ON_TIMEOUT,
                               WORKER_KILL,
                               JOB_FAILED,
                               WORKER_DONE], \
            "Aborting: Unknown calculation status received. " \
            "Received status: {}".format(calc_status)

    def _receive_from_workers(self, persis_info):
        """Receive calculation output from workers. Loops over all
        active workers and probes to see if worker is ready to
        communticate. If any output is received, all other workers are
        looped back over.
        """
        new_stuff = True
        while new_stuff and any(self.W['active']):
            new_stuff = False
            for w in self.W['worker_id'][self.W['active'] > 0]:
                if self.wcomms[w-1].mail_flag():
                    new_stuff = True
                    self._handle_msg_from_worker(persis_info, w)

        if 'save_every_k' in self.sim_specs:
            self._save_every_k_sims()
        if 'save_every_k' in self.gen_specs:
            self._save_every_k_gens()
        return persis_info

    def _update_state_on_worker_msg(self, persis_info, D_recv, w):
        """Update history and worker info on worker message.
        """
        calc_type = D_recv['calc_type']
        calc_status = D_recv['calc_status']
        Manager._check_received_calc(D_recv)

        self.W[w-1]['active'] = 0
        if calc_status in [FINISHED_PERSISTENT_SIM_TAG,
                           FINISHED_PERSISTENT_GEN_TAG]:
            self.W[w-1]['persis_state'] = 0
        else:
            if calc_type == EVAL_SIM_TAG:
                self.hist.update_history_f(D_recv)
            if calc_type == EVAL_GEN_TAG:
                self.hist.update_history_x_in(w, D_recv['calc_out'])
                assert len(D_recv['calc_out']) or np.any(self.W['active']), \
                    "Gen must return work when is is the only thing active."
            if 'libE_info' in D_recv and 'persistent' in D_recv['libE_info']:
                # Now a waiting, persistent worker
                self.W[w-1]['persis_state'] = calc_type

        if 'libE_info' in D_recv and 'blocking' in D_recv['libE_info']:
            # Now done blocking these workers
            for w_i in D_recv['libE_info']['blocking']:
                self.W[w_i-1]['blocked'] = 0
                self.W[w_i-1]['active'] = 0

        if 'persis_info' in D_recv and len(D_recv['persis_info']):
            persis_info[w].update(D_recv['persis_info'])

    def _handle_msg_from_worker(self, persis_info, w):
        """Handle a message from worker w.
        """
        logger.debug("Manager receiving from Worker: {}".format(w))
        try:
            msg = self.wcomms[w-1].recv()
            tag, D_recv = msg
        except CommFinishedException:
            logger.debug("Finalizing message from Worker {}".format(w))
            return

        if isinstance(D_recv, WorkerErrMsg):
            self.W[w-1]['active'] = 0
            self._kill_workers()
            raise ManagerException('Received error message from {}'.format(w),
                                   D_recv.msg, D_recv.exc)
        elif isinstance(D_recv, logging.LogRecord):
            logging.getLogger(D_recv.name).handle(D_recv)
        else:
            self._update_state_on_worker_msg(persis_info, D_recv, w)

    # --- Handle termination

    def _final_receive_and_kill(self, persis_info):
        """
        Tries to receive from any active workers.

        If time expires before all active workers have been received from, a
        nonblocking receive is posted (though the manager will not receive this
        data) and a kill signal is sent.
        """
        exit_flag = 0
        while any(self.W['active']) and exit_flag == 0:
            persis_info = self._receive_from_workers(persis_info)
            if self.term_test(logged=False) == 2 and any(self.W['active']):
                print(_WALLCLOCK_MSG)
                sys.stdout.flush()
                sys.stderr.flush()
                exit_flag = 2

        self._kill_workers()
        print("\nlibEnsemble manager total time:", self.elapsed())
        return persis_info, exit_flag

    # --- Main loop

    def _alloc_work(self, H, persis_info):
        "Call work allocation function from alloc_specs"
        alloc_f = self.alloc_specs['alloc_f']
        return alloc_f(self.W, H, self.sim_specs, self.gen_specs, self.alloc_specs, persis_info)

    def run(self, persis_info):
        "Run the manager."
        logger.info("Manager initiated on node {}".format(socket.gethostname()))
        logger.info("Manager exit_criteria: {}".format(self.exit_criteria))

        # Continue receiving and giving until termination test is satisfied
        try:
            while not self.term_test():
                persis_info = self._receive_from_workers(persis_info)
                if any(self.W['active'] == 0):
                    Work, persis_info = self._alloc_work(self.hist.trim_H(),
                                                         persis_info)
                    for w in Work:
                        if self.term_test():
                            break
                        self._check_work_order(Work[w], w)
                        self._send_work_order(Work[w], w)
                        self._update_state_on_alloc(Work[w], w)
                assert self.term_test() or any(self.W['active'] != 0), \
                    "Should not wait for workers when all workers are idle."

        finally:
            # Return persis_info, exit_flag
            result = self._final_receive_and_kill(persis_info)
        return result
