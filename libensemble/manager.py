"""
libEnsemble manager routines
============================
"""

import cProfile
import glob
import logging
import os
import platform
import socket
import sys
import traceback
from queue import SimpleQueue
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from numpy.lib.recfunctions import repack_fields

from libensemble.comms.comms import QComm
from libensemble.message_numbers import EVAL_GEN_TAG, EVAL_SIM_TAG, PERSIS_STOP, calc_status_strings
from libensemble.resources.resources import Resources
from libensemble.tools.fields_keys import protected_libE_fields
from libensemble.tools.tools import _USER_CALC_DIR_WARNING
from libensemble.utils.output_directory import EnsembleDirectory
from libensemble.utils.pipelines import ManagerFromWorker, ManagerToWorker
from libensemble.utils.timer import Timer

logger = logging.getLogger(__name__)
# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)


class ManagerException(Exception):
    """Exception raised by the Manager"""


class WorkerException(Exception):
    """Exception raised on abort signal from worker"""


class LoggedException(Exception):
    """Raise exception for handling without re-logging"""


def report_worker_exc(wrk_exc: Exception = None) -> None:
    """Write worker exception to log"""
    if wrk_exc is not None:
        from_line, msg, exc = wrk_exc.args
        logger.error(f"---- {from_line} ----")
        logger.error(f"Message: {msg}")
        logger.error(exc)


def manager_main(
    hist,
    libE_specs: dict,
    alloc_specs: dict,
    sim_specs: dict,
    gen_specs: dict,
    exit_criteria: dict,
    persis_info: dict,
    wcomms: list = [],
) -> (dict, int, int):
    """Manager routine to coordinate the generation and simulation evaluations

    Parameters
    ----------

    hist: :obj:`libensemble.history.History`
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

    wcomms: :obj:`list`, Optional
        A list of comm type objects for each worker. Default is an empty list.
    """
    if libE_specs.get("profile"):
        pr = cProfile.Profile()
        pr.enable()

    # Send dtypes to workers
    dtypes = {
        EVAL_SIM_TAG: repack_fields(hist.H[sim_specs["in"]]).dtype,
        EVAL_GEN_TAG: repack_fields(hist.H[gen_specs["in"]]).dtype,
    }

    for wcomm in wcomms:
        wcomm.send(0, dtypes)

    if libE_specs.get("use_workflow_dir"):
        for wcomm in wcomms:
            wcomm.send(0, libE_specs.get("workflow_dir_path"))

    libE_specs["_dtypes"] = dtypes

    # Set up and run manager
    mgr = Manager(hist, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, wcomms)
    result = mgr.run(persis_info)

    if libE_specs.get("profile"):
        pr.disable()
        profile_stats_fname = "manager.prof"
        pr.dump_stats(profile_stats_fname)

    return result


def filter_nans(array: npt.NDArray) -> npt.NDArray:
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

    def __init__(
        self,
        hist,
        libE_specs: dict,
        alloc_specs: dict,
        sim_specs: dict,
        gen_specs: dict,
        exit_criteria: dict,
        wcomms: list = [],
    ):
        """Initializes the manager"""
        timer = Timer()
        timer.start()
        self.date_start = timer.date_start.replace(" ", "_")
        self.safe_mode = libE_specs.get("safe_mode")
        self.kill_canceled_sims = libE_specs.get("kill_canceled_sims")
        self.hist = hist
        self.hist.safe_mode = self.safe_mode
        self.libE_specs = libE_specs
        self.alloc_specs = alloc_specs
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.exit_criteria = exit_criteria
        self.elapsed = lambda: timer.elapsed
        self.wcomms = wcomms
        self.WorkerExc = False
        self.persis_pending = []

        dyn_keys = ("resource_sets", "num_procs", "num_gpus")
        dyn_keys_in_H = any(k in self.hist.H.dtype.names for k in dyn_keys)
        self.use_resource_sets = dyn_keys_in_H or self.libE_specs.get("num_resource_sets")
        self.gen_num_procs = libE_specs.get("gen_num_procs", 0)
        self.gen_num_gpus = libE_specs.get("gen_num_gpus", 0)

        self.W = np.zeros(len(self.wcomms) + 1, dtype=Manager.worker_dtype)
        self.W["worker_id"] = np.arange(len(self.wcomms) + 1)
        self.term_tests = [
            (2, "wallclock_max", self.term_test_wallclock),
            (1, "sim_max", self.term_test_sim_max),
            (1, "gen_max", self.term_test_gen_max),
            (1, "stop_val", self.term_test_stop_val),
        ]

        self.self_inbox = SimpleQueue()
        self.self_outbox = SimpleQueue()

        self.wcomms = [QComm(self.self_inbox, self.self_outbox, len(self.W))] + self.wcomms

        temp_EnsembleDirectory = EnsembleDirectory(libE_specs=libE_specs)
        self.resources = Resources.resources
        self.scheduler_opts = self.libE_specs.get("scheduler_opts", {})
        if self.resources is not None:
            gresource = self.resources.glob_resources
            self.scheduler_opts = gresource.update_scheduler_opts(self.scheduler_opts)
            for wrk in self.W:
                if wrk["worker_id"] in gresource.zero_resource_workers:
                    wrk["zero_resource_worker"] = True

        try:
            temp_EnsembleDirectory.make_copyback()
        except AssertionError as e:  # Ensemble dir exists and isn't empty.
            logger.manager_warning(_USER_CALC_DIR_WARNING.format(temp_EnsembleDirectory.ensemble_dir))
            self._kill_workers()
            raise ManagerException(
                "Manager errored on initialization",
                "Ensemble directory already existed and wasn't empty.",
                "To reuse ensemble dir, set libE_specs['reuse_output_dir'] = True",
                e,
            )

    # --- Termination logic routines

    def term_test_wallclock(self, max_elapsed: int) -> bool:
        """Checks against wallclock timeout"""
        return self.elapsed() >= max_elapsed

    def term_test_sim_max(self, sim_max: int) -> bool:
        """Checks against max simulations"""
        return self.hist.sim_ended_count >= sim_max + self.hist.sim_ended_offset

    def term_test_gen_max(self, gen_max: int) -> bool:
        """Checks against max generator calls"""
        return self.hist.index >= gen_max + self.hist.gen_informed_offset

    def term_test_stop_val(self, stop_val: Any) -> bool:
        """Checks against stop value criterion"""
        key, val = stop_val
        H = self.hist.H
        return np.any(filter_nans(H[key][H["sim_ended"]]) <= val)

    def term_test(self, logged: bool = True) -> Union[bool, int]:
        """Checks termination criteria"""
        for retval, key, testf in self.term_tests:
            if key in self.exit_criteria:
                if testf(self.exit_criteria[key]):
                    if logged:
                        logger.info(f"Term test tripped: {key}")
                    return retval
        return 0

    # --- Checkpointing logic

    def _get_date_start_str(self) -> str:
        """Get timestamp for workflow start, for saving History"""
        date_start = self.date_start + "_"
        if platform.system() == "Windows":
            date_start = date_start.replace(":", "-")  # ":" is invalid in windows filenames
        if not self.libE_specs["save_H_with_date"]:
            date_start = ""
        return date_start

    def _save_every_k(self, fname: str, count: int, k: int, complete: bool) -> None:
        """Saves history every kth step"""
        if not complete:
            count = k * (count // k)
        date_start = self._get_date_start_str()

        filename = fname.format(self.libE_specs["H_file_prefix"], date_start, count)
        if (not os.path.isfile(filename) and count > 0) or complete:
            for old_file in glob.glob(fname.format(self.libE_specs["H_file_prefix"], date_start, "*")):
                os.remove(old_file)
            np.save(filename, self.hist.trim_H())

    def _save_every_k_sims(self, complete: bool) -> None:
        """Saves history every kth sim step"""
        self._save_every_k(
            os.path.join(self.libE_specs["workflow_dir_path"], "{}_{}after_sim_{}.npy"),
            self.hist.sim_ended_count,
            self.libE_specs["save_every_k_sims"],
            complete,
        )

    def _save_every_k_gens(self, complete: bool) -> None:
        """Saves history every kth gen step"""
        self._save_every_k(
            os.path.join(self.libE_specs["workflow_dir_path"], "{}_{}after_gen_{}.npy"),
            self.hist.index,
            self.libE_specs["save_every_k_gens"],
            complete,
        )

    def _init_every_k_save(self, complete=False) -> None:
        force_final = complete and not self.libE_specs.get("save_every_k_gens")
        if self.libE_specs.get("save_every_k_sims") or force_final:
            self._save_every_k_sims(complete)
        if self.libE_specs.get("save_every_k_gens"):
            self._save_every_k_gens(complete)

    # --- Handle incoming messages from workers

    @staticmethod
    def _check_received_calc(D_recv: dict) -> None:
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

    # --- Handle termination

    def _sim_max_given(self) -> bool:
        if "sim_max" in self.exit_criteria:
            return self.hist.sim_started_count >= self.exit_criteria["sim_max"] + self.hist.sim_started_offset
        else:
            return False

    def _get_alloc_libE_info(self) -> dict:
        """Selected statistics useful for alloc_f"""

        return {
            "any_idle_workers": any(self.W["active"] == 0),
            "exit_criteria": self.exit_criteria,
            "elapsed_time": self.elapsed(),
            "gen_informed_count": self.hist.gen_informed_count,
            "manager_kill_canceled_sims": self.kill_canceled_sims,
            "scheduler_opts": self.scheduler_opts,
            "sim_started_count": self.hist.sim_started_count,
            "sim_ended_count": self.hist.sim_ended_count,
            "sim_max_given": self._sim_max_given(),
            "use_resource_sets": self.use_resource_sets,
            "gen_num_procs": self.gen_num_procs,
            "gen_num_gpus": self.gen_num_gpus,
        }

    def _alloc_work(self, H: npt.NDArray, persis_info: dict) -> dict:
        """
        Calls work allocation function from alloc_specs. Copies protected libE
        fields before the alloc_f call and ensures they weren't modified
        """
        if self.safe_mode:
            saveH = repack_fields(H[protected_libE_fields], recurse=True)

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

    # --- Main loop

    def run(self, persis_info: dict) -> (dict, int, int):
        """Runs the manager"""
        logger.info(f"Manager initiated on node {socket.gethostname()}")
        logger.info(f"Manager exit_criteria: {self.exit_criteria}")

        self.ToWorker = ManagerToWorker(self)
        self.FromWorker = ManagerFromWorker(self)

        # Continue receiving and giving until termination test is satisfied
        try:
            while not self.term_test():
                self.ToWorker._kill_cancelled_sims()
                persis_info = self.FromWorker._receive_from_workers(persis_info)
                self._init_every_k_save()
                Work, persis_info, flag = self._alloc_work(self.hist.trim_H(), persis_info)
                if flag:
                    break

                for w in Work:
                    if self._sim_max_given():
                        break
                    self.ToWorker._check_work_order(Work[w], w)
                    self.ToWorker._send_work_order(Work[w], w)
                    self.ToWorker._update_state_on_alloc(Work[w], w)
                assert self.term_test() or any(
                    self.W["active"] != 0
                ), "alloc_f did not return any work, although all workers are idle."
        except WorkerException as e:  # catches all error messages from worker
            report_worker_exc(e)
            raise LoggedException(e.args[0], e.args[1]) from None
        except Exception as e:  # should only catch bugs within manager, or AssertionErrors
            logger.error(traceback.format_exc())
            raise LoggedException(e.args) from None
        finally:
            # Return persis_info, exit_flag, elapsed time
            result = self.FromWorker._final_receive_and_kill(persis_info)
            self._init_every_k_save(complete=self.libE_specs["save_H_on_completion"])
            sys.stdout.flush()
            sys.stderr.flush()
        return result
