import inspect
import logging
import logging.handlers
import time

import numpy.typing as npt

from libensemble.comms.comms import QCommThread
from libensemble.generators import LibensembleGenerator, PersistentGenInterfacer
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.utils.misc import list_dicts_to_np, np_to_list_dicts

logger = logging.getLogger(__name__)


class Runner:
    @classmethod
    def from_specs(cls, specs):
        if len(specs.get("globus_compute_endpoint", "")) > 0:
            return GlobusComputeRunner(specs)
        if specs.get("threaded"):
            return ThreadRunner(specs)
        if (generator := specs.get("generator")) is not None:
            if isinstance(generator, PersistentGenInterfacer):
                return LibensembleGenThreadRunner(specs)
            if isinstance(generator, LibensembleGenerator):
                return LibensembleGenRunner(specs)
            else:
                return StandardGenRunner(specs)
        else:
            return Runner(specs)

    def __init__(self, specs):
        self.specs = specs
        self.f = specs.get("sim_f") or specs.get("gen_f")

    def _truncate_args(self, calc_in: npt.NDArray, persis_info, libE_info):
        nparams = len(inspect.signature(self.f).parameters)
        args = [calc_in, persis_info, self.specs, libE_info]
        return args[:nparams]

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, int | None):
        """User function called in-place"""
        args = self._truncate_args(calc_in, persis_info, libE_info)
        return self.f(*args)

    def shutdown(self) -> None:
        pass

    def run(self, calc_in: npt.NDArray, Work: dict) -> (npt.NDArray, dict, int | None):
        if Work["persis_info"] is None:
            Work["persis_info"] = {}
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

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, int | None):
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

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, int | None):
        args = self._truncate_args(calc_in, persis_info, libE_info)
        self.thread_handle = QCommThread(self.f, None, *args, user_function=True)
        self.thread_handle.run()
        return self.thread_handle.result()

    def shutdown(self) -> None:
        if self.thread_handle is not None:
            self.thread_handle.terminate()


class StandardGenRunner(Runner):
    """Interact with suggest/ingest generator. Base class initialized for third-party generators."""

    def __init__(self, specs):
        super().__init__(specs)
        self.gen = specs.get("generator")

    def _get_points_updates(self, batch_size: int) -> (npt.NDArray, npt.NDArray):
        # no suggest_updates on external gens
        return (
            list_dicts_to_np(
                self.gen.suggest(batch_size), dtype=self.specs.get("out"), mapping=self.gen.variables_mapping
            ),
            None,
        )

    def _convert_ingest(self, x: npt.NDArray) -> list:
        self.gen.ingest(np_to_list_dicts(x))

    def _loop_over_gen(self, tag, Work, H_in):
        """Interact with suggest/ingest generator that *does not* contain a background thread"""
        while tag not in [PERSIS_STOP, STOP_TAG]:
            batch_size = self.specs.get("batch_size") or len(H_in)
            H_out, _ = self._get_points_updates(batch_size)
            tag, Work, H_in = self.ps.send_recv(H_out)
            self._convert_ingest(H_in)
        return H_in

    def _get_initial_suggest(self, libE_info) -> npt.NDArray:
        """Get initial batch from generator based on generator type"""
        initial_batch = self.specs.get("initial_batch_size") or self.specs.get("batch_size") or libE_info["batch_size"]
        H_out = self.gen.suggest(initial_batch)
        return H_out

    def _start_generator_loop(self, tag, Work, H_in):
        """Start the generator loop after choosing best way of giving initial results to gen"""
        self.gen.ingest(np_to_list_dicts(H_in))
        return self._loop_over_gen(tag, Work, H_in)

    def _persistent_result(self, calc_in, persis_info, libE_info):
        """Setup comms with manager, setup gen, loop gen to completion, return gen's results"""
        self.ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
        # libE gens will hit the following line, but list_dicts_to_np will passthrough if the output is a numpy array
        H_out = list_dicts_to_np(
            self._get_initial_suggest(libE_info), dtype=self.specs.get("out"), mapping=self.gen.variables_mapping
        )
        tag, Work, H_in = self.ps.send_recv(H_out)  # evaluate the initial sample
        final_H_in = self._start_generator_loop(tag, Work, H_in)
        return self.gen.finalize(final_H_in), FINISHED_PERSISTENT_GEN_TAG

    def _result(self, calc_in: npt.NDArray, persis_info: dict, libE_info: dict) -> (npt.NDArray, dict, int):
        if libE_info.get("persistent"):
            return self._persistent_result(calc_in, persis_info, libE_info)
        raise ValueError(
            "suggest/ingest generators must run in persistent mode. This may be the default in the future."
        )


class LibensembleGenRunner(StandardGenRunner):
    def _get_initial_suggest(self, libE_info) -> npt.NDArray:
        """Get initial batch from generator based on generator type"""
        H_out = self.gen.suggest_numpy(libE_info["batch_size"])  # OR GEN SPECS INITIAL BATCH SIZE
        return H_out

    def _get_points_updates(self, batch_size: int) -> (npt.NDArray, list):
        numpy_out = self.gen.suggest_numpy(batch_size)
        if callable(getattr(self.gen, "suggest_updates", None)):
            updates = self.gen.suggest_updates()
        else:
            updates = None
        return numpy_out, updates

    def _convert_ingest(self, x: npt.NDArray) -> list:
        self.gen.ingest_numpy(x)

    def _start_generator_loop(self, tag, Work, H_in) -> npt.NDArray:
        """Start the generator loop after choosing best way of giving initial results to gen"""
        self.gen.ingest_numpy(H_in)
        return self._loop_over_gen(tag, Work, H_in)  # see parent class


class LibensembleGenThreadRunner(StandardGenRunner):
    def _get_initial_suggest(self, libE_info) -> npt.NDArray:
        """Get initial batch from generator based on generator type"""
        return self.gen.suggest_numpy()  # libE really needs to receive the *entire* initial batch from a threaded gen

    def _suggest_and_send(self):
        """Loop over generator's outbox contents, send to manager"""
        while not self.gen.running_gen_f.outbox.empty():  # recv/send any outstanding messages
            points = self.gen.suggest_numpy()
            if callable(getattr(self.gen, "suggest_updates", None)):
                updates = self.gen.suggest_updates()
            else:
                updates = None
            if updates is not None and len(updates):
                self.ps.send(points)
                for i in updates:
                    self.ps.send(i, keep_state=True)  # keep_state since an update doesn't imply "new points"
            else:
                self.ps.send(points)

    def _loop_over_gen(self, *args):
        """Cycle between moving all outbound / inbound messages between threaded gen and manager"""
        while True:
            time.sleep(0.0025)  # dont need to ping the gen relentlessly. Let it calculate. 400hz
            self._suggest_and_send()
            while self.ps.comm.mail_flag():  # receive any new messages from Manager, give all to gen
                tag, _, H_in = self.ps.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    return H_in  # this will get inserted into finalize. this breaks loop
                self.gen.ingest_numpy(H_in)
