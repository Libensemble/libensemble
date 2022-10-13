"""
libEnsemble manager routines
============================
"""

import sys
import os
import glob
import logging
import socket
import platform
import traceback
import numpy as np

from libensemble.utils.timer import Timer
from libensemble.utils.misc import extract_H_ranges

from libensemble.message_numbers import (
    EVAL_SIM_TAG,
    EVAL_GEN_TAG,
    PERSIS_STOP,
    STOP_TAG,
    MAN_SIGNAL_FINISH,
    MAN_SIGNAL_KILL,
    FINISHED_PERSISTENT_SIM_TAG,
    FINISHED_PERSISTENT_GEN_TAG,
    calc_status_strings,
)

from libensemble.message_numbers import calc_type_strings
from libensemble.comms.comms import CommFinishedException
from libensemble.worker import WorkerErrMsg
from libensemble.output_directory import EnsembleDirectory
from libensemble.tools.tools import _USER_CALC_DIR_WARNING
from libensemble.resources.resources import Resources
from libensemble.tools.tools import _PERSIS_RETURN_WARNING
from libensemble.tools.fields_keys import protected_libE_fields
import cProfile
import pstats
import copy
import time

if tuple(np.__version__.split(".")) >= ("1", "15"):
    from numpy.lib.recfunctions import repack_fields

logger = logging.getLogger(__name__)
# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)


class ManagerException(Exception):
    """Exception raised by the Manager"""


class WorkerException(Exception):
    """Exception raised on abort signal from worker"""


class LoggedException(Exception):
    """Raise exception for handling without re-logging"""


def report_worker_exc(wrk_exc=None):
    """Write worker exception to log"""
    if wrk_exc is not None:
        from_line, msg, exc = wrk_exc.args
        logger.error(f"---- {from_line} ----")
        logger.error(f"Message: {msg}")
        logger.error(exc)


def manager_main(hist, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, persis_info, wcomms=[]):
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
    if libE_specs.get("profile"):
        pr = cProfile.Profile()
        pr.enable()

    if "in" not in gen_specs:
        gen_specs["in"] = []

    # Send dtypes to workers
    if "repack_fields" in globals():
        dtypes = {
            EVAL_SIM_TAG: repack_fields(hist.H[sim_specs["in"]]).dtype,
            EVAL_GEN_TAG: repack_fields(hist.H[gen_specs["in"]]).dtype,
        }
    else:
        dtypes = {
            EVAL_SIM_TAG: hist.H[sim_specs["in"]].dtype,
            EVAL_GEN_TAG: hist.H[gen_specs["in"]].dtype,
        }

    for wcomm in wcomms:
        wcomm.send(0, dtypes)

    # Set up and run manager
    mgr = Manager(hist, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, wcomms)
    result = mgr.run(persis_info)

    if libE_specs.get("profile"):
        pr.disable()
        profile_stats_fname = "manager.prof"

        with open(profile_stats_fname, "w") as f:
            ps = pstats.Stats(pr, stream=f).sort_stats("cumulative")
            ps.print_stats()

    return result


def filter_nans(array):
    """Filters out NaNs from a numpy array"""
    return array[~np.isnan(array)]


_WALLCLOCK_MSG_ALL_RETURNED = """
Termination due to wallclock_max has occurred.
All completed work has been returned.
Posting kill messages for all workers.
"""

_WALLCLOCK_MSG_ACTIVE = """
Termination due to wallclock_max has occurred.
Some issued work has not been returned.
Posting kill messages for all workers.
"""


class Manager:
    """Manager class for libensemble."""

    worker_dtype = [
        ("worker_id", int),
        ("active", int),
        ("persis_state", int),
        ("active_recv", int),
        ("gen_started_time", float),
        ("zero_resource_worker", bool),
    ]

    def __init__(self, hist, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, wcomms=[]):
        """Initializes the manager"""
        timer = Timer()
        timer.start()
        self.date_start = timer.date_start.replace(" ", "_")
        self.safe_mode = libE_specs.get("safe_mode", True)
        self.kill_canceled_sims = libE_specs.get("kill_canceled_sims", True)
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
        self.W["worker_id"] = np.arange(len(self.wcomms)) + 1
        self.term_tests = [
            (2, "wallclock_max", self.term_test_wallclock),
            (1, "sim_max", self.term_test_sim_max),
            (1, "gen_max", self.term_test_gen_max),
            (1, "stop_val", self.term_test_stop_val),
        ]

        temp_EnsembleDirectory = EnsembleDirectory(libE_specs=libE_specs)
        self.resources = Resources.resources
        if self.resources is not None:
            for wrk in self.W:
                if wrk["worker_id"] in self.resources.glob_resources.zero_resource_workers:
                    wrk["zero_resource_worker"] = True

        try:
            temp_EnsembleDirectory.make_copyback_check()
        except OSError as e:  # Ensemble dir exists and isn't empty.
            logger.manager_warning(_USER_CALC_DIR_WARNING.format(temp_EnsembleDirectory.prefix))
            self._kill_workers()
            raise ManagerException(
                "Manager errored on initialization",
                "Ensemble directory already existed and wasn't empty.",
                e,
            )

    # --- Termination logic routines

    def term_test_wallclock(self, max_elapsed):
        """Checks against wallclock timeout"""
        return self.elapsed() >= max_elapsed

    def term_test_sim_max(self, sim_max):
        """Checks against max simulations"""
        return self.hist.sim_ended_count >= sim_max + self.hist.sim_ended_offset

    def term_test_gen_max(self, gen_max):
        """Checks against max generator calls"""
        return self.hist.index >= gen_max + self.hist.gen_informed_offset

    def term_test_stop_val(self, stop_val):
        """Checks against stop value criterion"""
        key, val = stop_val
        H = self.hist.H
        return np.any(filter_nans(H[key][H["sim_ended"]]) <= val)

    def term_test(self, logged=True):
        """Checks termination criteria"""
        for retval, key, testf in self.term_tests:
            if key in self.exit_criteria:
                if testf(self.exit_criteria[key]):
                    if logged:
                        logger.info(f"Term test tripped: {key}")
                    return retval
        return 0

    # --- Low-level communication routines

    def _kill_workers(self):
        """Kills the workers"""
        for w in self.W["worker_id"]:
            self.wcomms[w - 1].send(STOP_TAG, MAN_SIGNAL_FINISH)

    # --- Checkpointing logic

    def _save_every_k(self, fname, count, k):
        """Saves history every kth step"""
        count = k * (count // k)
        filename = fname.format(self.date_start, count)
        if platform.system() == "Windows":
            filename = filename.replace(":", "-")  # ":" is invalid in windows filenames
        if not os.path.isfile(filename) and count > 0:
            for old_file in glob.glob(fname.format(self.date_start, "*")):
                os.remove(old_file)
            np.save(filename, self.hist.H)

    def _save_every_k_sims(self):
        """Saves history every kth sim step"""
        self._save_every_k(
            "libE_history_for_run_starting_{}_after_sim_{}.npy",
            self.hist.sim_ended_count,
            self.libE_specs["save_every_k_sims"],
        )

    def _save_every_k_gens(self):
        """Saves history every kth gen step"""
        self._save_every_k(
            "libE_history_for_run_starting_{}_after_gen_{}.npy",
            self.hist.index,
            self.libE_specs["save_every_k_gens"],
        )

    # --- Handle outgoing messages to workers (work orders from alloc)

    def _check_work_order(self, Work, w):
        """Checks validity of an allocation function order"""
        assert w != 0, "Can't send to worker 0; this is the manager."
        if self.W[w - 1]["active_recv"]:
            assert "active_recv" in Work["libE_info"], (
                "Messages to a worker in active_recv mode should have active_recv"
                f"set to True in libE_info. Work['libE_info'] is {Work['libE_info']}"
            )
        else:
            assert self.W[w - 1]["active"] == 0, (
                "Allocation function requested work be sent to worker %d, an already active worker." % w
            )
        work_rows = Work["libE_info"]["H_rows"]
        if len(work_rows):
            work_fields = set(Work["H_fields"])

            assert len(work_fields), (
                f"Allocation function requested rows={work_rows} be sent to worker={w}, "
                "but requested no fields to be sent."
            )
            hist_fields = self.hist.H.dtype.names
            diff_fields = list(work_fields.difference(hist_fields))

            assert not diff_fields, f"Allocation function requested invalid fields {diff_fields} be sent to worker={w}."

    def _set_resources(self, Work, w):
        """Check rsets given in Work match rsets assigned in resources.

        If rsets are not assigned, then assign using default mapping
        """
        resource_manager = self.resources.resource_manager
        rset_req = Work["libE_info"].get("rset_team")

        if rset_req is None:
            rset_team = []
            default_rset = resource_manager.index_list[w - 1]
            if default_rset is not None:
                rset_team.append(default_rset)
            Work["libE_info"]["rset_team"] = rset_team

        resource_manager.assign_rsets(Work["libE_info"]["rset_team"], w)

    def _freeup_resources(self, w):
        """Free up resources assigned to the worker"""
        if self.resources:
            self.resources.resource_manager.free_rsets(w)

    def _send_work_order(self, Work, w):
        """Sends an allocation function order to a worker"""
        logger.debug(f"Manager sending work unit to worker {w}")

        if self.resources:
            self._set_resources(Work, w)

        self.wcomms[w - 1].send(Work["tag"], Work)

        if Work["tag"] == EVAL_GEN_TAG:
            self.W[w - 1]["gen_started_time"] = time.time()

        work_rows = Work["libE_info"]["H_rows"]
        work_name = calc_type_strings[Work["tag"]]
        logger.debug(f"Manager sending {work_name} work to worker {w}. Rows {extract_H_ranges(Work) or None}")
        if len(work_rows):
            if "repack_fields" in globals():
                new_dtype = [(name, self.hist.H.dtype.fields[name][0]) for name in Work["H_fields"]]
                H_to_be_sent = np.empty(len(work_rows), dtype=new_dtype)
                for i, row in enumerate(work_rows):
                    H_to_be_sent[i] = repack_fields(self.hist.H[Work["H_fields"]][row])
                # H_to_be_sent = repack_fields(self.hist.H[Work['H_fields']])[work_rows]
                self.wcomms[w - 1].send(0, H_to_be_sent)
            else:
                self.wcomms[w - 1].send(0, self.hist.H[Work["H_fields"]][work_rows])

    def _update_state_on_alloc(self, Work, w):
        """Updates a workers' active/idle status following an allocation order"""
        self.W[w - 1]["active"] = Work["tag"]
        if "libE_info" in Work:
            if "persistent" in Work["libE_info"]:
                self.W[w - 1]["persis_state"] = Work["tag"]
                if Work["libE_info"].get("active_recv", False):
                    self.W[w - 1]["active_recv"] = Work["tag"]
            else:
                assert "active_recv" not in Work["libE_info"], "active_recv worker must also be persistent"

        work_rows = Work["libE_info"]["H_rows"]
        if Work["tag"] == EVAL_SIM_TAG:
            self.hist.update_history_x_out(work_rows, w)
        elif Work["tag"] == EVAL_GEN_TAG:
            self.hist.update_history_to_gen(work_rows)

    # --- Handle incoming messages from workers

    @staticmethod
    def _check_received_calc(D_recv):
        """Checks the type and status fields on a receive calculation"""
        calc_type = D_recv["calc_type"]
        calc_status = D_recv["calc_status"]
        assert calc_type in [
            EVAL_SIM_TAG,
            EVAL_GEN_TAG,
        ], f"Aborting, Unknown calculation type received. Received type: {calc_type}"

        assert calc_status in list(calc_status_strings.keys()) + [PERSIS_STOP] or isinstance(
            calc_status, str
        ), f"Aborting: Unknown calculation status received. Received status: {calc_status}"

    def _receive_from_workers(self, persis_info):
        """Receives calculation output from workers. Loops over all
        active workers and probes to see if worker is ready to
        communticate. If any output is received, all other workers are
        looped back over.
        """
        new_stuff = True
        while new_stuff:
            new_stuff = False
            for w in self.W["worker_id"]:
                if self.wcomms[w - 1].mail_flag():
                    new_stuff = True
                    self._handle_msg_from_worker(persis_info, w)

        if "save_every_k_sims" in self.libE_specs:
            self._save_every_k_sims()
        if "save_every_k_gens" in self.libE_specs:
            self._save_every_k_gens()
        return persis_info

    def _update_state_on_worker_msg(self, persis_info, D_recv, w):
        """Updates history and worker info on worker message"""
        calc_type = D_recv["calc_type"]
        calc_status = D_recv["calc_status"]
        Manager._check_received_calc(D_recv)

        keep_state = D_recv["libE_info"].get("keep_state", False)
        if w not in self.persis_pending and not self.W[w - 1]["active_recv"] and not keep_state:
            self.W[w - 1]["active"] = 0

        if calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG]:
            final_data = D_recv.get("calc_out", None)
            if isinstance(final_data, np.ndarray):
                if calc_status is FINISHED_PERSISTENT_GEN_TAG and self.libE_specs.get("use_persis_return_gen", False):
                    self.hist.update_history_x_in(w, final_data, self.safe_mode, self.W[w - 1]["gen_started_time"])
                elif calc_status is FINISHED_PERSISTENT_SIM_TAG and self.libE_specs.get("use_persis_return_sim", False):
                    self.hist.update_history_f(D_recv, self.safe_mode)
                else:
                    logger.info(_PERSIS_RETURN_WARNING)
            self.W[w - 1]["persis_state"] = 0
            if self.W[w - 1]["active_recv"]:
                self.W[w - 1]["active"] = 0
                self.W[w - 1]["active_recv"] = 0
            if w in self.persis_pending:
                self.persis_pending.remove(w)
                self.W[w - 1]["active"] = 0
            self._freeup_resources(w)
        else:
            if calc_type == EVAL_SIM_TAG:
                self.hist.update_history_f(D_recv, self.safe_mode)
            if calc_type == EVAL_GEN_TAG:
                self.hist.update_history_x_in(w, D_recv["calc_out"], self.safe_mode, self.W[w - 1]["gen_started_time"])
                assert (
                    len(D_recv["calc_out"]) or np.any(self.W["active"]) or self.W[w - 1]["persis_state"]
                ), "Gen must return work when is is the only thing active and not persistent."
            if "libE_info" in D_recv and "persistent" in D_recv["libE_info"]:
                # Now a waiting, persistent worker
                self.W[w - 1]["persis_state"] = calc_type
            else:
                self._freeup_resources(w)

        if D_recv.get("persis_info"):
            persis_info[w].update(D_recv["persis_info"])

    def _handle_msg_from_worker(self, persis_info, w):
        """Handles a message from worker w"""
        try:
            msg = self.wcomms[w - 1].recv()
            tag, D_recv = msg
        except CommFinishedException:
            logger.debug(f"Finalizing message from Worker {w}")
            return
        if isinstance(D_recv, WorkerErrMsg):
            self.W[w - 1]["active"] = 0
            logger.debug(f"Manager received exception from worker {w}")
            if not self.WorkerExc:
                self.WorkerExc = True
                self._kill_workers()
                raise WorkerException(f"Received error message from worker {w}", D_recv.msg, D_recv.exc)
        elif isinstance(D_recv, logging.LogRecord):
            logger.debug(f"Manager received a log message from worker {w}")
            logging.getLogger(D_recv.name).handle(D_recv)
        else:
            logger.debug(f"Manager received data message from worker {w}")
            self._update_state_on_worker_msg(persis_info, D_recv, w)

    def _kill_cancelled_sims(self):
        """Send kill signals to any sims marked as cancel_requested"""
        if self.kill_canceled_sims:
            kill_sim = (
                self.hist.H["sim_started"]
                & self.hist.H["cancel_requested"]
                & ~self.hist.H["sim_ended"]
                & ~self.hist.H["kill_sent"]
            )

            # Note that a return is still expected when running sims are killed
            if np.any(kill_sim):
                logger.debug(f"Manager sending kill signals to H indices {np.where(kill_sim)}")
                kill_ids = self.hist.H["sim_id"][kill_sim]
                kill_on_workers = self.hist.H["sim_worker"][kill_sim]
                for w in kill_on_workers:
                    self.wcomms[w - 1].send(STOP_TAG, MAN_SIGNAL_KILL)
                    self.hist.H["kill_sent"][kill_ids] = True

    # --- Handle termination

    def _final_receive_and_kill(self, persis_info):
        """
        Tries to receive from any active workers.

        If time expires before all active workers have been received from, a
        nonblocking receive is posted (though the manager will not receive this
        data) and a kill signal is sent.
        """

        # Send a handshake signal to each persistent worker.
        if any(self.W["persis_state"]):
            for w in self.W["worker_id"][self.W["persis_state"] > 0]:
                logger.debug(f"Manager sending PERSIS_STOP to worker {w}")
                if "final_fields" in self.libE_specs:
                    rows_to_send = self.hist.trim_H()["sim_ended"]
                    fields_to_send = self.libE_specs["final_fields"]
                    H_to_send = self.hist.trim_H()[rows_to_send][fields_to_send]
                    self.wcomms[w - 1].send(PERSIS_STOP, H_to_send)
                else:
                    self.wcomms[w - 1].send(PERSIS_STOP, MAN_SIGNAL_KILL)
                if not self.W[w - 1]["active"]:
                    # Re-activate if necessary
                    self.W[w - 1]["active"] = self.W[w - 1]["persis_state"]
                self.persis_pending.append(w)

        exit_flag = 0
        while (any(self.W["active"]) or any(self.W["persis_state"])) and exit_flag == 0:
            persis_info = self._receive_from_workers(persis_info)
            if self.term_test(logged=False) == 2:
                # Elapsed Wallclock has expired
                if not any(self.W["persis_state"]):
                    if any(self.W["active"]):
                        logger.manager_warning(_WALLCLOCK_MSG_ACTIVE)
                    else:
                        logger.manager_warning(_WALLCLOCK_MSG_ALL_RETURNED)
                    exit_flag = 2
            if self.WorkerExc:
                exit_flag = 1

        self._kill_workers()
        return persis_info, exit_flag, self.elapsed()

    # --- Main loop

    def _sim_max_given(self):
        if "sim_max" in self.exit_criteria:
            return self.hist.sim_started_count >= self.exit_criteria["sim_max"] + self.hist.sim_started_offset
        else:
            return False

    def _get_alloc_libE_info(self):
        """Selected statistics useful for alloc_f"""
        return {
            "any_idle_workers": any(self.W["active"] == 0),
            "exit_criteria": self.exit_criteria,
            "elapsed_time": self.elapsed(),
            "gen_informed_count": self.hist.gen_informed_count,
            "manager_kill_canceled_sims": self.kill_canceled_sims,
            "scheduler_opts": self.libE_specs.get("scheduler_opts", {}),
            "sim_started_count": self.hist.sim_started_count,
            "sim_ended_count": self.hist.sim_ended_count,
            "sim_max_given": self._sim_max_given(),
            "use_resource_sets": "num_resource_sets" in self.libE_specs,
        }

    def _alloc_work(self, H, persis_info):
        """
        Calls work allocation function from alloc_specs. Copies protected libE
        fields before the alloc_f call and ensures they weren't modified
        """
        if self.safe_mode:
            if "repack_fields" in globals():
                saveH = repack_fields(H[protected_libE_fields], recurse=True)
            else:
                saveH = copy.deepcopy(H[protected_libE_fields])

        alloc_f = self.alloc_specs["alloc_f"]
        output = alloc_f(
            self.W,
            H,
            self.sim_specs,
            self.gen_specs,
            self.alloc_specs,
            persis_info,
            self._get_alloc_libE_info(),
        )

        if self.safe_mode:
            assert np.array_equal(saveH, H[protected_libE_fields]), "The allocation function modified protected fields"

        if len(output) == 2:
            output = output + ((0,))

        assert len(output) == 3, "alloc_f must return three outputs."
        assert isinstance(output[0], dict), "First alloc_f output must be a dictionary"
        assert isinstance(output[1], dict), "Second alloc_f output must be a dictionary"
        assert output[2] in [0, 1], "Third alloc_f output must be 0 or 1."

        return output

    def run(self, persis_info):
        """Runs the manager"""
        logger.info(f"Manager initiated on node {socket.gethostname()}")
        logger.info(f"Manager exit_criteria: {self.exit_criteria}")

        # Continue receiving and giving until termination test is satisfied
        try:
            while not self.term_test():
                self._kill_cancelled_sims()
                persis_info = self._receive_from_workers(persis_info)
                Work, persis_info, flag = self._alloc_work(self.hist.trim_H(), persis_info)
                if flag:
                    break

                for w in Work:
                    if self._sim_max_given():
                        break
                    self._check_work_order(Work[w], w)
                    self._send_work_order(Work[w], w)
                    self._update_state_on_alloc(Work[w], w)
                assert self.term_test() or any(
                    self.W["active"] != 0
                ), "alloc_f did not return any work, although all workers are idle."
        except WorkerException as e:
            report_worker_exc(e)
            raise LoggedException(e.args[0], e.args[1]) from None
        except Exception as e:
            logger.error(traceback.format_exc())
            raise LoggedException(e.args) from None
        finally:
            # Return persis_info, exit_flag, elapsed time
            result = self._final_receive_and_kill(persis_info)
            sys.stdout.flush()
            sys.stderr.flush()
        return result
