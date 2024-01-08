import inspect
import logging
import logging.handlers
from typing import Callable, Optional

import numpy.typing as npt

logger = logging.getLogger(__name__)


class Runner:
    def __new__(cls, specs):
        if len(specs.get("globus_compute_endpoint", "")) > 0:
            return super(Runner, GlobusComputeRunner).__new__(GlobusComputeRunner)
        if specs.get("threaded"):  # TODO: undecided interface
            return super(Runner, ThreadRunner).__new__(ThreadRunner)
        else:
            return Runner

    def __init__(self, specs):
        self.specs = specs
        self.f = specs.get("sim_f") or specs.get("gen_f")

    def _truncate_args(self, calc_in, persis_info, specs, libE_info, user_f):
        nparams = len(inspect.signature(user_f).parameters)
        args = [calc_in, persis_info, specs, libE_info]
        return args[:nparams]

    def _result(
        self, calc_in: npt.NDArray, persis_info: dict, specs: dict, libE_info: dict, user_f: Callable, tag: int
    ) -> (npt.NDArray, dict, Optional[int]):
        """User function called in-place"""
        args = self._truncate_args(calc_in, persis_info, specs, libE_info, user_f)
        return user_f(*args)

    def shutdown(self) -> None:
        pass

    def run(self, calc_in, Work):
        return self._result(calc_in, Work["persis_info"], self.specs, Work["libE_info"], self.f, Work["tag"])


class GlobusComputeRunner(Runner):
    def __init__(self, specs):
        super().__init__(specs)
        self.globus_compute_executor = self._get_globus_compute_executor()(endpoint_id=specs["globus_compute_endpoint"])
        self.globus_compute_fid = self.globus_compute_executor.register_function(self.f)

    def shutdown(self) -> None:
        self.globus_compute_executor.shutdown()

    def _get_globus_compute_executor(self):
        try:
            from globus_compute_sdk import Executor
        except ModuleNotFoundError:
            logger.warning("Globus Compute use detected but Globus Compute not importable. Is it installed?")
            logger.warning("Running function evaluations normally on local resources.")
            return None
        else:
            return Executor

    def _result(
        self, calc_in: npt.NDArray, persis_info: dict, specs: dict, libE_info: dict, user_f: Callable, tag: int
    ) -> (npt.NDArray, dict, Optional[int]):
        from libensemble.worker import Worker

        libE_info["comm"] = None  # 'comm' object not pickle-able
        Worker._set_executor(0, None)  # ditto for executor

        fargs = self._truncate_args(calc_in, persis_info, specs, libE_info, user_f)
        exctr = self.globus_compute_executor
        func_id = self.globus_compute_fid

        task_fut = exctr.submit_to_registered_function(func_id, fargs)
        return task_fut.result()


class ThreadRunner(Runner):
    pass
