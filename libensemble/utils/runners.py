import inspect
import logging
import logging.handlers
import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from libensemble.comms.comms import QCommThread
from libensemble.generators import LibensembleGenerator, LibensembleGenThreadInterfacer, np_to_list_dicts
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

logger = logging.getLogger(__name__)


class Runner:
    def __new__(cls, specs):
        if len(specs.get("globus_compute_endpoint", "")) > 0:
            return super(Runner, GlobusComputeRunner).__new__(GlobusComputeRunner)
        if specs.get("threaded"):  # TODO: undecided interface
            return super(Runner, ThreadRunner).__new__(ThreadRunner)
        if isinstance(specs.get("generator", None), LibensembleGenThreadInterfacer):
            return super(AskTellGenRunner, LibensembleGenThreadInterfacer).__new__(LibensembleGenThreadInterfacer)
        if isinstance(specs.get("generator", None), LibensembleGenerator):
            return super(AskTellGenRunner, LibensembleGenRunner).__new__(LibensembleGenRunner)
        if hasattr(specs.get("generator", None), "ask"):  # all other ask/tell gens, third party
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
    """Interact with ask/tell generator. Base class initialized for third-party generators."""

    def __init__(self, specs):
        super().__init__(specs)
        self.gen = specs.get("generator")
        self.inital_batch = getattr(self.gen, "initial_batch_size", 0)
        self.batch = getattr(self.gen, "batch_size", 0)

    def _to_array(self, x: list) -> npt.NDArray:
        """fast-cast list-of-dicts to NumPy array"""
        if isinstance(x, list) and len(x) and isinstance(x[0], dict):
            arr = np.zeros(len(x), dtype=self.specs["out"])
            for i in range(len(x)):
                for key in x[0].keys():
                    arr[i][key] = x[i][key]
            return arr
        return x

    def _loop_over_gen(self, tag, Work):
        """Interact with ask/tell generator that *does not* contain a background thread"""
        while tag not in [PERSIS_STOP, STOP_TAG]:
            batch_size = self.batch or Work["libE_info"]["batch_size"]
            points, updates = self._to_array(self.gen.ask(batch_size)), self._to_array(self.gen.ask_updates())
            if updates is not None and len(updates):  # returned "samples" and "updates". can combine if same dtype
                H_out = np.append(points, updates)
            else:
                H_out = points
            tag, Work, H_in = self.ps.send_recv(H_out)
            self.gen.tell(np_to_list_dicts(H_in))
        return H_in

    def _get_initial_ask(self, libE_info) -> npt.NDArray:
        """Get initial batch from generator based on generator type"""
        initial_batch = self.inital_batch or libE_info["batch_size"]
        H_out = self.gen.ask(initial_batch)
        return H_out

    def _start_generator_loop(self, tag, Work, H_in):
        """Start the generator loop after choosing best way of giving initial results to gen"""
        self.gen.tell(np_to_list_dicts(H_in))
        final_H_in = self._loop_over_gen(tag, Work)
        return final_H_in

    def _persistent_result(self, calc_in, persis_info, libE_info):
        """Setup comms with manager, setup gen, loop gen to completion, return gen's results"""
        self.ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
        if hasattr(self.gen, "setup"):
            self.gen.persis_info = persis_info  # passthrough, setup() uses the gen attributes
            self.gen.libE_info = libE_info
            if self.gen.thread is None:
                self.gen.setup()  # maybe we're reusing a live gen from a previous run
        H_out = self._get_initial_ask(libE_info)
        tag, Work, H_in = self.ps.send_recv(H_out)  # evaluate the initial sample
        final_H_in = self._start_generator_loop(tag, Work, H_in)
        return self.gen.final_tell(final_H_in), FINISHED_PERSISTENT_GEN_TAG

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, Optional[int]):
        if libE_info.get("persistent"):
            return self._persistent_result(calc_in, persis_info, libE_info)
        return self._to_array(self.gen.ask(getattr(self.gen, "batch_size", 0) or libE_info["batch_size"]))


class LibensembleGenRunner(AskTellGenRunner):
    def _get_initial_ask(self, libE_info) -> npt.NDArray:
        """Get initial batch from generator based on generator type"""
        H_out = self.gen.ask_np(self.inital_batch or libE_info["batch_size"])
        return H_out

    def _start_generator_loop(self, tag, Work, H_in) -> npt.NDArray:
        """Start the generator loop after choosing best way of giving initial results to gen"""
        self.gen.tell_np(H_in)
        return self._loop_over_libe_asktell_gen(tag, Work)

    def _loop_over_libe_asktell_gen(self, tag, Work) -> npt.NDArray:
        """Interact with ask/tell generator that *does not* contain a background thread"""
        while tag not in [PERSIS_STOP, STOP_TAG]:
            batch_size = self.batch or Work["libE_info"]["batch_size"]
            points, updates = self.gen.ask_np(batch_size), self.gen.ask_updates()
            if updates is not None and len(updates):  # returned "samples" and "updates". can combine if same dtype
                H_out = np.append(points, updates)
            else:
                H_out = points
            tag, Work, H_in = self.ps.send_recv(H_out)
            self.gen.tell_np(H_in)
        return H_in


class LibensembleGenThreadRunner(AskTellGenRunner):
    def _get_initial_ask(self, libE_info) -> npt.NDArray:
        """Get initial batch from generator based on generator type"""
        H_out = self.gen.ask_np()  # libE really needs to receive the *entire* initial batch from a threaded gen
        return H_out

    def _start_generator_loop(self, tag, Work, H_in):
        """Start the generator loop after choosing best way of giving initial results to gen"""
        self.gen.tell_np(H_in)
        return self._loop_over_thread_interfacer()

    def _ask_and_send(self):
        """Loop over generator's outbox contents, send to manager"""
        while self.gen.outbox.qsize():  # recv/send any outstanding messages
            points, updates = self.gen.ask_np(), self.gen.ask_updates()
            if updates is not None and len(updates):
                self.ps.send(points)
                for i in updates:
                    self.ps.send(i, keep_state=True)  # keep_state since an update doesn't imply "new points"
            else:
                self.ps.send(points)

    def _loop_over_thread_interfacer(self):
        """Cycle between moving all outbound / inbound messages between threaded gen and manager"""
        while True:
            time.sleep(0.0025)  # dont need to ping the gen relentlessly. Let it calculate. 400hz
            self._ask_and_send()
            while self.ps.comm.mail_flag():  # receive any new messages from Manager, give all to gen
                tag, _, H_in = self.ps.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    return H_in  # this will get inserted into final_tell. this breaks loop
                self.gen.tell_np(H_in)
