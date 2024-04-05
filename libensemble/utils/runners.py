import inspect
import logging
import logging.handlers
import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from libensemble.comms.comms import QCommThread
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

logger = logging.getLogger(__name__)


class Runner:
    def __new__(cls, specs):
        if len(specs.get("globus_compute_endpoint", "")) > 0:
            return super(Runner, GlobusComputeRunner).__new__(GlobusComputeRunner)
        if specs.get("threaded"):  # TODO: undecided interface
            return super(Runner, ThreadRunner).__new__(ThreadRunner)
        if hasattr(specs.get("generator", None), "ask"):
            return super(Runner, AskTellGenRunner).__new__(AskTellGenRunner)
        else:
            return super().__new__(Runner)

    def __init__(self, specs):
        self.specs = specs
        self.f = specs.get("sim_f") or specs.get("gen_f")

    def _truncate_args(self, calc_in: npt.NDArray, persis_info, libE_info):
        nparams = len(inspect.signature(self.f).parameters)
        args = [calc_in, persis_info, self.specs, libE_info]
        return args[:nparams]

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, Optional[int]):
        """User function called in-place"""
        args = self._truncate_args(calc_in, persis_info, libE_info)
        return self.f(*args)

    def shutdown(self) -> None:
        pass

    def run(self, calc_in: npt.NDArray, Work: dict) -> (npt.NDArray, dict, Optional[int]):
        return self._result(calc_in, Work["persis_info"], Work["libE_info"])


class GlobusComputeRunner(Runner):
    def __init__(self, specs):
        super().__init__(specs)
        self.globus_compute_executor = self._get_globus_compute_executor()(endpoint_id=specs["globus_compute_endpoint"])
        self.globus_compute_fid = self.globus_compute_executor.register_function(self.f)

    def _get_globus_compute_executor(self):
        try:
            from globus_compute_sdk import Executor
        except ModuleNotFoundError:
            logger.warning("Globus Compute use detected but Globus Compute not importable. Is it installed?")
            logger.warning("Running function evaluations normally on local resources.")
            return None
        else:
            return Executor

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, Optional[int]):
        from libensemble.worker import Worker

        libE_info["comm"] = None  # 'comm' object not pickle-able
        Worker._set_executor(0, None)  # ditto for executor

        args = self._truncate_args(calc_in, persis_info, libE_info)
        task_fut = self.globus_compute_executor.submit_to_registered_function(self.globus_compute_fid, args)
        return task_fut.result()

    def shutdown(self) -> None:
        self.globus_compute_executor.shutdown()


class ThreadRunner(Runner):
    def __init__(self, specs):
        super().__init__(specs)
        self.thread_handle = None

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, Optional[int]):
        args = self._truncate_args(calc_in, persis_info, libE_info)
        self.thread_handle = QCommThread(self.f, None, *args, user_function=True)
        self.thread_handle.run()
        return self.thread_handle.result()

    def shutdown(self) -> None:
        if self.thread_handle is not None:
            self.thread_handle.terminate()


class AskTellGenRunner(Runner):
    def __init__(self, specs):
        super().__init__(specs)
        self.gen = specs.get("generator")

    def _persistent_result(self, calc_in, persis_info, libE_info):
        self.ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
        tag = None
        initial_batch = getattr(self.gen, "initial_batch_size", 0) or libE_info["batch_size"]
        if hasattr(self.gen, "setup"):
            self.gen.persis_info = persis_info
            self.gen.libE_info = libE_info
            self.gen.setup()
        H_out = self.gen.initial_ask(initial_batch, calc_in)
        tag, Work, H_in = self.ps.send_recv(H_out)  # evaluate the initial sample
        self.gen.tell(H_in)  # tell the gen the initial sample results
        batch_size = getattr(self.gen, "batch_size", 0) or Work["libE_info"]["batch_size"]
        STOP = False
        while not STOP:
            time.sleep(0.0025)  # dont need to ping the gen relentlessly. Let it calculate. 400hz
            for _ in range(self.gen.outbox.qsize()):  # send any outstanding messages
                points = self.gen.ask(batch_size)
                if len(points) == 2:  # returned "samples" and "updates". can combine if same dtype
                    H_out = np.append(points[0], points[1])
                else:
                    H_out = points
                self.ps.send(H_out)
            while self.ps.comm.mail_flag():  # receive any new messages, give all to gen
                tag, _, H_in = self.ps.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    STOP = True
                    break
                self.gen.tell(H_in)
        return self.gen.final_tell(H_in), FINISHED_PERSISTENT_GEN_TAG

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, Optional[int]):
        if libE_info.get("persistent"):
            return self._persistent_result(calc_in, persis_info, libE_info)
        return self.gen.ask(getattr(self.gen, "batch_size", 0) or libE_info["batch_size"])
