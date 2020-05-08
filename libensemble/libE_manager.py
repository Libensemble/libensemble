"""
libEnsemble manager routines
============================
"""

import sys
import os
import glob
import logging
import socket
import numpy as np

from libensemble.utils.timer import Timer
from libensemble.message_numbers import \
    EVAL_SIM_TAG, FINISHED_PERSISTENT_SIM_TAG, \
    EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, \
    STOP_TAG, UNSET_TAG, PERSIS_STOP, \
    WORKER_KILL, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, \
    TASK_FAILED, WORKER_DONE, \
    MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL
from libensemble.comms.comms import CommFinishedException
from libensemble.libE_worker import WorkerErrMsg
from libensemble.tools.tools import _USER_SIM_DIR_WARNING
from libensemble.tools.fields_keys import libE_spec_calc_dir_keys
import cProfile
import pstats

logger = logging.getLogger(__name__)
# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)


class ManagerException(Exception):
    "Exception at manager, raised on abort signal from worker"


def manager_main(hist, libE_specs, alloc_specs,
                 sim_specs, gen_specs, exit_criteria, persis_info, wcomms=[]):
    """Manager routine to coordinate the generation and simulation evaluations

    Parameters
    ----------

    hist: :obj:`History`
        A libEnsemble history type object.

    libE_specs: :obj:`dict`
        Specifications for libEnsemble

    alloc_specs: :obj:`dict`
        Specifications for the allocation function

    sim_specs: :obj:`dict`
        Specifications for the simulation function

    gen_specs: :obj:`dict`
        Specifications for the generator function

    exit_criteria: :obj:`dict`
        Criteria for libEnsemble to stop a run

    persis_info: :obj:`dict`
        Persistent information to be passed between user functions

    wcomms: :obj:`list`, optional
        A list of comm type objects for each worker. Default is an empty list.
    """
    if sim_specs.get('profile'):
        pr = cProfile.Profile()
        pr.enable()

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
    result = mgr.run(persis_info)

    if sim_specs.get('profile'):
        pr.disable()
        profile_stats_fname = 'manager.prof'

        with open(profile_stats_fname, 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_stats()

    return result


def filter_nans(array):
    "Filters out NaNs from a numpy array"
    return array[~np.isnan(array)]


_WALLCLOCK_MSG_ALL_RETURNED = """
Termination due to elapsed_wallclock_time has occurred.
All completed work has been returned.
Posting kill messages for all workers.
"""

_WALLCLOCK_MSG_ACTIVE = """
Termination due to elapsed_wallclock_time has occurred.
Some issued work has not been returned.
Posting kill messages for all workers.
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
        """Initializes the manager"""
        timer = Timer()
        timer.start()
        self.date_start = timer.date_start.replace(' ', '_')
        self.hist = hist
        self.libE_specs = libE_specs
        self.alloc_specs = alloc_specs
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.exit_criteria = exit_criteria
        self.elapsed = lambda: timer.elapsed
        self.wcomms = wcomms
        self.WorkerExc = False
        self.persis_pending = []
        self.W = np.zeros(len(self.wcomms), dtype=Manager.worker_dtype)
        self.W['worker_id'] = np.arange(len(self.wcomms)) + 1
        self.term_tests = \
            [(2, 'elapsed_wallclock_time', self.term_test_wallclock),
             (1, 'sim_max', self.term_test_sim_max),
             (1, 'gen_max', self.term_test_gen_max),
             (1, 'stop_val', self.term_test_stop_val)]

        if any([setting in self.libE_specs for setting in libE_spec_calc_dir_keys]):
            self.check_ensemble_dir(libE_specs)
            if libE_specs.get('ensemble_copy_back', False):
                Manager.make_copyback_dir(libE_specs)

    @staticmethod
    def make_copyback_dir(libE_specs):
        ensemble_dir_path = libE_specs.get('ensemble_dir_path', './ensemble')
        copybackdir = os.path.basename(ensemble_dir_path)  # Current directory, same basename
        if os.path.relpath(ensemble_dir_path) == os.path.relpath(copybackdir):
            copybackdir += '_back'
        os.makedirs(copybackdir)

    def check_ensemble_dir(self, libE_specs):
        prefix = libE_specs.get('ensemble_dir_path', './ensemble')
        try:
            os.rmdir(prefix)
        except FileNotFoundError:  # Ensemble dir doesn't exist.
            pass
        except OSError as e:  # Ensemble dir exists and isn't empty.
            logger.manager_warning(_USER_SIM_DIR_WARNING.format(prefix))
            self._kill_workers()
            raise ManagerException('Manager errored on initialization',
                                   'Ensemble directory already existed and wasn\'t empty.', e)

    # --- Termination logic routines

    def term_test_wallclock(self, max_elapsed):
        """Checks against wallclock timeout"""
        return self.elapsed() >= max_elapsed

    def term_test_sim_max(self, sim_max):
        """Checks against max simulations"""
        return self.hist.given_count >= sim_max + self.hist.offset

    def term_test_gen_max(self, gen_max):
        """Checks against max generator calls"""
        return self.hist.index >= gen_max + self.hist.offset

    def term_test_stop_val(self, stop_val):
        """Checks against stop value criterion"""
        key, val = stop_val
        H = self.hist.H
        return np.any(filter_nans(H[key][H['returned']]) <= val)

    def term_test(self, logged=True):
        """Checks termination criteria"""
        for retval, key, testf in self.term_tests:
            if key in self.exit_criteria:
                if testf(self.exit_criteria[key]):
                    if logged:
                        logger.info("Term test tripped: {}".format(key))
                    return retval
        return 0

    # --- Low-level communication routines

    def _kill_workers(self):
        """Kills the workers"""
        for w in self.W['worker_id']:
            self.wcomms[w-1].send(STOP_TAG, MAN_SIGNAL_FINISH)

    # --- Checkpointing logic

    def _save_every_k(self, fname, count, k):
        "Saves history every kth step"
        count = k*(count//k)
        filename = fname.format(self.date_start, count)
        if not os.path.isfile(filename) and count > 0:
            for old_file in glob.glob(fname.format(self.date_start, '*')):
                os.remove(old_file)
            np.save(filename, self.hist.H)

    def _save_every_k_sims(self):
        "Saves history every kth sim step"
        self._save_every_k('libE_history_for_run_starting_{}_after_sim_{}.npy',
                           self.hist.sim_count,
                           self.libE_specs['save_every_k_sims'])

    def _save_every_k_gens(self):
        "Saves history every kth gen step"
        self._save_every_k('libE_history_for_run_starting_{}_after_gen_{}.npy',
                           self.hist.index,
                           self.libE_specs['save_every_k_gens'])

    # --- Handle outgoing messages to workers (work orders from alloc)

    def _check_work_order(self, Work, w):
        """Checks validity of an allocation function order
        """
        assert w != 0, "Can't send to worker 0; this is the manager."
        assert self.W[w-1]['active'] == 0, \
            "Allocation function requested work be sent to to worker %d, an "\
            "already active worker." % w
        work_rows = Work['libE_info']['H_rows']
        if len(work_rows):
            work_fields = set(Work['H_fields'])
            assert len(work_fields), \
                "Allocation function requested rows={} be sent to worker={}, "\
                "but requested no fields to be sent.".format(work_rows, w)
            hist_fields = self.hist.H.dtype.names
            diff_fields = list(work_fields.difference(hist_fields))
            assert not diff_fields, \
                "Allocation function requested invalid fields {}" \
                "be sent to worker={}.".format(diff_fields, w)

    def _send_work_order(self, Work, w):
        """Sends an allocation function order to a worker
        """
        logger.debug("Manager sending work unit to worker {}".format(w))
        self.wcomms[w-1].send(Work['tag'], Work)
        work_rows = Work['libE_info']['H_rows']
        if len(work_rows):
            self.wcomms[w-1].send(0, self.hist.H[Work['H_fields']][work_rows])

    def _update_state_on_alloc(self, Work, w):
        """Updates a workers' active/idle status following an allocation order"""

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
        "Checks the type and status fields on a receive calculation"
        calc_type = D_recv['calc_type']
        calc_status = D_recv['calc_status']
        assert calc_type in [EVAL_SIM_TAG, EVAL_GEN_TAG], \
            "Aborting, Unknown calculation type received. " \
            "Received type: {}".format(calc_type)
        assert calc_status in [FINISHED_PERSISTENT_SIM_TAG,
                               FINISHED_PERSISTENT_GEN_TAG,
                               UNSET_TAG,
                               PERSIS_STOP,
                               MAN_SIGNAL_FINISH,
                               MAN_SIGNAL_KILL,
                               WORKER_KILL_ON_ERR,
                               WORKER_KILL_ON_TIMEOUT,
                               WORKER_KILL,
                               TASK_FAILED,
                               WORKER_DONE], \
            "Aborting: Unknown calculation status received. " \
            "Received status: {}".format(calc_status)

    def _receive_from_workers(self, persis_info):
        """Receives calculation output from workers. Loops over all
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

        if 'save_every_k_sims' in self.libE_specs:
            self._save_every_k_sims()
        if 'save_every_k_gens' in self.libE_specs:
            self._save_every_k_gens()
        return persis_info

    def _update_state_on_worker_msg(self, persis_info, D_recv, w):
        """Updates history and worker info on worker message
        """
        calc_type = D_recv['calc_type']
        calc_status = D_recv['calc_status']
        Manager._check_received_calc(D_recv)

        if w not in self.persis_pending:
            self.W[w-1]['active'] = 0

        if calc_status in [FINISHED_PERSISTENT_SIM_TAG,
                           FINISHED_PERSISTENT_GEN_TAG]:
            self.W[w-1]['persis_state'] = 0
            if w in self.persis_pending:
                self.persis_pending.remove(w)
                self.W[w-1]['active'] = 0
        else:
            if calc_type == EVAL_SIM_TAG:
                self.hist.update_history_f(D_recv)
            if calc_type == EVAL_GEN_TAG:
                self.hist.update_history_x_in(w, D_recv['calc_out'])
                assert len(D_recv['calc_out']) or np.any(self.W['active']) or self.W[w-1]['persis_state'], \
                    "Gen must return work when is is the only thing active and not persistent."
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
        """Handles a message from worker w
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
            if not self.WorkerExc:
                self.WorkerExc = True
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

        # Send a handshake signal to each persistent worker.
        if any(self.W['persis_state']):
            for w in self.W['worker_id'][self.W['persis_state'] > 0]:
                logger.debug("Manager sending PERSIS_STOP to worker {}".format(w))
                self.wcomms[w-1].send(PERSIS_STOP, MAN_SIGNAL_KILL)
                if not self.W[w-1]['active']:
                    # Re-activate if necessary
                    self.W[w-1]['active'] = self.W[w-1]['persis_state']
                self.persis_pending.append(w)

        exit_flag = 0
        while (any(self.W['active']) or any(self.W['persis_state'])) and exit_flag == 0:
            persis_info = self._receive_from_workers(persis_info)
            if self.term_test(logged=False) == 2:
                # Elapsed Wallclock has expired
                if not any(self.W['persis_state']):
                    if any(self.W['active']):
                        logger.manager_warning(_WALLCLOCK_MSG_ACTIVE)
                    else:
                        logger.manager_warning(_WALLCLOCK_MSG_ALL_RETURNED)
                    exit_flag = 2
            if self.WorkerExc:
                exit_flag = 1

        self._kill_workers()
        return persis_info, exit_flag, self.elapsed()

    # --- Main loop

    def _alloc_work(self, H, persis_info):
        "Calls work allocation function from alloc_specs"
        alloc_f = self.alloc_specs['alloc_f']
        output = alloc_f(self.W, H, self.sim_specs, self.gen_specs, self.alloc_specs, persis_info)

        if len(output) == 2:
            output = output + ((0,))

        assert len(output) == 3, "alloc_f must return three outputs."
        assert isinstance(output[0], dict), "First alloc_f output must be a dictionary"
        assert isinstance(output[1], dict), "Second alloc_f output must be a dictionary"
        assert output[2] in [0, 1], "Third alloc_f output must be 0 or 1."

        return output

    def run(self, persis_info):
        "Runs the manager"
        logger.info("Manager initiated on node {}".format(socket.gethostname()))
        logger.info("Manager exit_criteria: {}".format(self.exit_criteria))

        # Continue receiving and giving until termination test is satisfied
        try:
            while not self.term_test():
                persis_info = self._receive_from_workers(persis_info)
                if any(self.W['active'] == 0):
                    Work, persis_info, flag = self._alloc_work(self.hist.trim_H(),
                                                               persis_info)
                    if flag:
                        break

                    for w in Work:
                        if self.term_test():
                            break
                        self._check_work_order(Work[w], w)
                        self._send_work_order(Work[w], w)
                        self._update_state_on_alloc(Work[w], w)
                assert self.term_test() or any(self.W['active'] != 0), \
                    "alloc_f did not return any work, although all workers are idle."

        finally:
            # Return persis_info, exit_flag, elapsed time
            result = self._final_receive_and_kill(persis_info)
            sys.stdout.flush()
            sys.stderr.flush()
        return result
