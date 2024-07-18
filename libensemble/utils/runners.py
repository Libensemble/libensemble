import inspect
import logging
import logging.handlers
import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from libensemble.comms.comms import QCommThread
from libensemble.generators import LibensembleGenerator, LibEnsembleGenInterfacer, np_to_list_dicts
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

    def _to_array(self, x):
        if isinstance(x, list) and len(x) and isinstance(x[0], dict):
            arr = np.zeros(len(x), dtype=self.specs["out"])
            for i in range(len(x)):
                for key in x[0].keys():
                    arr[i][key] = x[i][key]
            return arr
        return x

    def _loop_over_normal_generator(self, tag, Work):
        while tag not in [PERSIS_STOP, STOP_TAG]:
            batch_size = getattr(self.gen, "batch_size", 0) or Work["libE_info"]["batch_size"]
            if issubclass(type(self.gen), LibensembleGenerator):
                points, updates = self.gen.ask_np(batch_size), self.gen.ask_updates()
            else:
                points, updates = self._to_array(self.gen.ask(batch_size)), self._to_array(self.gen.ask_updates())
            if updates is not None and len(updates):  # returned "samples" and "updates". can combine if same dtype
                H_out = np.append(points, updates)
            else:
                H_out = points
            tag, Work, H_in = self.ps.send_recv(H_out)
            if issubclass(type(self.gen), LibensembleGenerator):
                self.gen.tell_np(H_in)
            else:
                self.gen.tell(np_to_list_dicts(H_in))
        return H_in

    def _ask_and_send(self):
        while self.gen.outbox.qsize():  # recv/send any outstanding messages
            points, updates = self.gen.ask_np(), self.gen.ask_updates()  # PersistentInterfacers each have ask_np
            if updates is not None and len(updates):
                self.ps.send(points)
                for i in updates:
                    self.ps.send(i, keep_state=True)
            else:
                self.ps.send(points)

    def _loop_over_persistent_interfacer(self):
        while True:
            time.sleep(0.0025)  # dont need to ping the gen relentlessly. Let it calculate. 400hz
            self._ask_and_send()
            while self.ps.comm.mail_flag():  # receive any new messages, give all to gen
                tag, _, H_in = self.ps.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    return H_in
                self.gen.tell_np(H_in)

    def _persistent_result(self, calc_in, persis_info, libE_info):
        self.ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
        tag = None
        if hasattr(self.gen, "setup"):
            self.gen.persis_info = persis_info  # passthrough, setup() uses the gen attributes
            self.gen.libE_info = libE_info
            if self.gen.thread is None:
                self.gen.setup()  # maybe we're reusing a live gen from a previous run
        initial_batch = getattr(self.gen, "initial_batch_size", 0) or libE_info["batch_size"]
        if issubclass(
            type(self.gen), LibEnsembleGenInterfacer
        ):  # we can't control how many points created by a threaded gen
            H_out = self.gen.ask_np()  # updates can probably be ignored when asking the first time
        elif issubclass(type(self.gen), LibensembleGenerator):
            H_out = self.gen.ask_np(initial_batch)  # libE really needs to receive the *entire* initial batch
        else:
            H_out = self.gen.ask(initial_batch)
        tag, Work, H_in = self.ps.send_recv(H_out)  # evaluate the initial sample
        if issubclass(type(self.gen), LibEnsembleGenInterfacer):  # libE native-gens can ask/tell numpy arrays
            self.gen.tell_np(H_in)
            final_H_in = self._loop_over_persistent_interfacer()
        elif issubclass(type(self.gen), LibensembleGenerator):
            self.gen.tell_np(H_in)
            final_H_in = self._loop_over_normal_generator(tag, Work)
        else:  # non-native gen, needs list of dicts
            self.gen.tell(np_to_list_dicts(H_in))
            final_H_in = self._loop_over_normal_generator(tag, Work)
        return self.gen.final_tell(final_H_in), FINISHED_PERSISTENT_GEN_TAG

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, Optional[int]):
        if libE_info.get("persistent"):
            return self._persistent_result(calc_in, persis_info, libE_info)
        return self._to_array(self.gen.ask(getattr(self.gen, "batch_size", 0) or libE_info["batch_size"]))
