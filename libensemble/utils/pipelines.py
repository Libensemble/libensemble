import logging
import time

import numpy as np
import numpy.typing as npt
from numpy.lib.recfunctions import repack_fields

from libensemble.comms.comms import CommFinishedException
from libensemble.message_numbers import (
    EVAL_GEN_TAG,
    EVAL_SIM_TAG,
    FINISHED_PERSISTENT_GEN_TAG,
    FINISHED_PERSISTENT_SIM_TAG,
    MAN_SIGNAL_FINISH,
    MAN_SIGNAL_KILL,
    PERSIS_STOP,
    STOP_TAG,
    calc_type_strings,
)
from libensemble.resources.resources import Resources
from libensemble.tools.tools import _PERSIS_RETURN_WARNING
from libensemble.utils.misc import extract_H_ranges
from libensemble.worker import WorkerErrMsg

logger = logging.getLogger(__name__)

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


class WorkerException(Exception):
    """Exception raised on abort signal from worker"""


class _WorkPipeline:
    def __init__(self, libE_specs, sim_specs, gen_specs):
        self.libE_specs = libE_specs
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs


class WorkerToManager(_WorkPipeline):
    def __init__(self, libE_specs, sim_specs, gen_specs):
        super().__init__(libE_specs, sim_specs, gen_specs)


class Worker:
    """Wrapper class for Worker array and worker comms"""

    def __init__(self, W: npt.NDArray, wid: int, wcomms: list = []):
        self.__dict__["_W"] = W
        self.__dict__["_wididx"] = wid - 1
        self.__dict__["_wcomms"] = wcomms

    def __setattr__(self, field, value):
        self._W[self._wididx][field] = value

    def __getattr__(self, field):
        return self._W[self._wididx][field]

    def update_state_on_alloc(self, Work: dict):
        self.active = Work["tag"]
        if "persistent" in Work["libE_info"]:
            self.persis_state = Work["tag"]
            if Work["libE_info"].get("active_recv", False):
                self.active_recv = Work["tag"]
        else:
            assert "active_recv" not in Work["libE_info"], "active_recv worker must also be persistent"

    def update_persistent_state(self):
        self.persis_state = 0
        if self.active_recv:
            self.active = 0
            self.active_recv = 0

    def set_work(self, Work):
        self.__dict__["_Work"] = Work

    def send(self, tag, data):
        self._wcomms[self._wididx].send(tag, data)

    def mail_flag(self):
        return self._wcomms[self._wididx].mail_flag()

    def recv(self):
        return self._wcomms[self._wididx].recv()


class _ManagerPipeline(_WorkPipeline):
    def __init__(self, Manager):
        super().__init__(Manager.libE_specs, Manager.sim_specs, Manager.gen_specs)
        self.W = Manager.W
        self.hist = Manager.hist
        self.wcomms = Manager.wcomms
        self.kill_canceled_sims = Manager.kill_canceled_sims
        self.persis_pending = Manager.persis_pending

    def _update_state_on_alloc(self, Work: dict, w: int):
        """Updates a workers' active/idle status following an allocation order"""
        worker = Worker(self.W, w, self.wcomms)
        worker.update_state_on_alloc(Work)

        work_rows = Work["libE_info"]["H_rows"]
        if Work["tag"] == EVAL_SIM_TAG:
            self.hist.update_history_x_out(work_rows, w, self.kill_canceled_sims)
        elif Work["tag"] == EVAL_GEN_TAG:
            self.hist.update_history_to_gen(work_rows)

    def _kill_workers(self) -> None:
        """Kills the workers"""
        for w in self.W["worker_id"]:
            self.wcomms[w - 1].send(STOP_TAG, MAN_SIGNAL_FINISH)


class ManagerFromWorker(_ManagerPipeline):
    def __init__(self, Manager):
        super().__init__(Manager)
        self.WorkerExc = False
        self.resources = Manager.resources
        self.term_test = Manager.term_test
        self.elapsed = Manager.elapsed

    def _handle_msg_from_worker(self, persis_info: dict, w: int) -> None:
        """Handles a message from worker w"""
        worker = Worker(self.W, w, self.wcomms)
        try:
            msg = worker.recv()
            _, D_recv = msg
        except CommFinishedException:
            logger.debug(f"Finalizing message from Worker {w}")
            return
        if isinstance(D_recv, WorkerErrMsg):
            worker.active = 0
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

    def _update_state_on_worker_msg(self, persis_info: dict, D_recv: dict, w: int) -> None:
        """Updates history and worker info on worker message"""
        calc_type = D_recv["calc_type"]
        calc_status = D_recv["calc_status"]

        worker = Worker(self.W, w, self.wcomms)

        keep_state = D_recv["libE_info"].get("keep_state", False)
        if w not in self.persis_pending and not worker.active_recv and not keep_state:
            worker.active = 0

        if calc_status in [FINISHED_PERSISTENT_SIM_TAG, FINISHED_PERSISTENT_GEN_TAG]:
            final_data = D_recv.get("calc_out", None)
            if isinstance(final_data, np.ndarray):
                if calc_status is FINISHED_PERSISTENT_GEN_TAG and self.libE_specs.get("use_persis_return_gen", False):
                    self.hist.update_history_x_in(w, final_data, worker.gen_started_time)
                elif calc_status is FINISHED_PERSISTENT_SIM_TAG and self.libE_specs.get("use_persis_return_sim", False):
                    self.hist.update_history_f(D_recv, self.kill_canceled_sims)
                else:
                    logger.info(_PERSIS_RETURN_WARNING)
            worker.update_persistent_state()
            if w in self.persis_pending:
                self.persis_pending.remove(w)
                worker.active = 0
            self._freeup_resources(w)
        else:
            if calc_type == EVAL_SIM_TAG:
                self.hist.update_history_f(D_recv, self.kill_canceled_sims)
            if calc_type == EVAL_GEN_TAG:
                self.hist.update_history_x_in(w, D_recv["calc_out"], worker.gen_started_time)
                assert (
                    len(D_recv["calc_out"]) or np.any(self.W["active"]) or worker.persis_state
                ), "Gen must return work when is is the only thing active and not persistent."
            if "libE_info" in D_recv and "persistent" in D_recv["libE_info"]:
                # Now a waiting, persistent worker
                worker.persis_state = calc_type
            else:
                self._freeup_resources(w)

    def _receive_from_workers(self, persis_info: dict) -> dict:
        """Receives calculation output from workers. Loops over all
        active workers and probes to see if worker is ready to
        communicate. If any output is received, all other workers are
        looped back over.
        """
        time.sleep(0.0001)  # Critical for multiprocessing performance
        new_stuff = True
        while new_stuff:
            new_stuff = False
            for w in self.W["worker_id"]:
                if self.wcomms[w - 1].mail_flag():
                    new_stuff = True
                    self._handle_msg_from_worker(persis_info, w)

        return persis_info

    def _final_receive_and_kill(self, persis_info: dict) -> (dict, int, int):
        """
        Tries to receive from any active workers.

        If time expires before all active workers have been received from, a
        nonblocking receive is posted (though the manager will not receive this
        data) and a kill signal is sent.
        """

        # Send a handshake signal to each persistent worker.
        if any(self.W["persis_state"]):
            for w in self.W["worker_id"][self.W["persis_state"] > 0]:
                worker = Worker(self.W, w, self.wcomms)
                logger.debug(f"Manager sending PERSIS_STOP to worker {w}")
                if self.libE_specs.get("final_gen_send", False):
                    rows_to_send = np.where(self.hist.H["sim_ended"] & ~self.hist.H["gen_informed"])[0]
                    work = {
                        "H_fields": self.gen_specs["persis_in"],
                        "persis_info": persis_info[w],
                        "tag": PERSIS_STOP,
                        "libE_info": {"persistent": True, "H_rows": rows_to_send},
                    }
                    self._send_work_order(work, w)
                    self.hist.update_history_to_gen(rows_to_send)
                else:
                    worker.send(PERSIS_STOP, MAN_SIGNAL_KILL)
                if not worker.active:
                    # Re-activate if necessary
                    worker.active = worker.persis_state
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

    def _freeup_resources(self, w: int) -> None:
        """Free up resources assigned to the worker"""
        if self.resources:
            self.resources.resource_manager.free_rsets(w)


class ManagerToWorker(_ManagerPipeline):
    def __init__(self, Manager):
        super().__init__(Manager)

    def _kill_cancelled_sims(self) -> None:
        """Send kill signals to any sims marked as cancel_requested"""

        if self.kill_canceled_sims:
            inds_to_check = np.arange(self.hist.last_ended + 1, self.hist.last_started + 1)

            kill_sim = (
                self.hist.H["sim_started"][inds_to_check]
                & self.hist.H["cancel_requested"][inds_to_check]
                & ~self.hist.H["sim_ended"][inds_to_check]
                & ~self.hist.H["kill_sent"][inds_to_check]
            )
            kill_sim_rows = inds_to_check[kill_sim]

            # Note that a return is still expected when running sims are killed
            if np.any(kill_sim):
                logger.debug(f"Manager sending kill signals to H indices {kill_sim_rows}")
                kill_ids = self.hist.H["sim_id"][kill_sim_rows]
                kill_on_workers = self.hist.H["sim_worker"][kill_sim_rows]
                for w in kill_on_workers:
                    self.wcomms[w - 1].send(STOP_TAG, MAN_SIGNAL_KILL)
                    self.hist.H["kill_sent"][kill_ids] = True

    @staticmethod
    def _set_resources(Work: dict, w: int) -> None:
        """Check rsets given in Work match rsets assigned in resources.

        If rsets are not assigned, then assign using default mapping
        """
        resource_manager = Resources.resources.resource_manager
        rset_req = Work["libE_info"].get("rset_team")

        if rset_req is None:
            rset_team = []
            default_rset = resource_manager.index_list[w - 1]
            if default_rset is not None:
                rset_team.append(default_rset)
            Work["libE_info"]["rset_team"] = rset_team

        resource_manager.assign_rsets(Work["libE_info"]["rset_team"], w)

    def _send_work_order(self, Work: dict, w: int) -> None:
        """Sends an allocation function order to a worker"""
        logger.debug(f"Manager sending work unit to worker {w}")

        worker = Worker(self.W, w, self.wcomms)

        if Resources.resources:
            self._set_resources(Work, w)

        worker.send(Work["tag"], Work)

        if Work["tag"] == EVAL_GEN_TAG:
            worker.gen_started_time = time.time()

        work_rows = Work["libE_info"]["H_rows"]
        work_name = calc_type_strings[Work["tag"]]
        logger.debug(f"Manager sending {work_name} work to worker {w}. Rows {extract_H_ranges(Work) or None}")
        if len(work_rows):
            new_dtype = [(name, self.hist.H.dtype.fields[name][0]) for name in Work["H_fields"]]
            H_to_be_sent = np.empty(len(work_rows), dtype=new_dtype)
            for i, row in enumerate(work_rows):
                H_to_be_sent[i] = repack_fields(self.hist.H[Work["H_fields"]][row])
            worker.send(0, H_to_be_sent)

    def _check_work_order(self, Work: dict, w: int, force: bool = False) -> None:
        """Checks validity of an allocation function order"""

        worker = Worker(self.W, w, self.wcomms)

        if worker.active_recv:
            assert "active_recv" in Work["libE_info"], (
                "Messages to a worker in active_recv mode should have active_recv"
                f"set to True in libE_info. Work['libE_info'] is {Work['libE_info']}"
            )
        else:
            if not force:
                assert worker.active == 0, (
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


class ManagerInplace(_ManagerPipeline):
    def __init__(self, libE_specs, sim_specs, gen_specs):
        super().__init__(libE_specs, sim_specs, gen_specs)
