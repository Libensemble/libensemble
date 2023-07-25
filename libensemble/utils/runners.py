import inspect
import logging
import logging.handlers
from typing import Callable, Dict, Optional

import numpy.typing as npt

from libensemble.message_numbers import EVAL_GEN_TAG, EVAL_SIM_TAG

logger = logging.getLogger(__name__)


class Runners:
    """Determines and returns methods for workers to run user functions.

    Currently supported: direct-call and Globus Compute
    """

    def __init__(self, sim_specs: dict, gen_specs: dict) -> None:
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.sim_f = sim_specs["sim_f"]
        self.gen_f = gen_specs.get("gen_f")
        self.has_globus_compute_sim = len(sim_specs.get("globus_compute_endpoint", "")) > 0
        self.has_globus_compute_gen = len(gen_specs.get("globus_compute_endpoint", "")) > 0

        if any([self.has_globus_compute_sim, self.has_globus_compute_gen]):
            if self.has_globus_compute_sim:
                self.sim_globus_compute_executor = self._get_globus_compute_executor()(
                    endpoint_id=self.sim_specs["globus_compute_endpoint"]
                )
                self.globus_compute_simfid = self.sim_globus_compute_executor.register_function(self.sim_f)

            if self.has_globus_compute_gen:
                self.gen_globus_compute_executor = self._get_globus_compute_executor()(
                    endpoint_id=self.gen_specs["globus_compute_endpoint"]
                )
                self.globus_compute_genfid = self.gen_globus_compute_executor.register_function(self.gen_f)

    def make_runners(self) -> Dict[int, Callable]:
        """Creates functions to run a sim or gen. These functions are either
        called directly by the worker or submitted to a Globus Compute endpoint."""

        def run_sim(calc_in, Work):
            """Determines how to run sim."""
            if self.has_globus_compute_sim:
                result = self._globus_compute_result
            else:
                result = self._normal_result

            return result(calc_in, Work["persis_info"], self.sim_specs, Work["libE_info"], self.sim_f, Work["tag"])

        if self.gen_specs:

            def run_gen(calc_in, Work):
                """Determines how to run gen."""
                if self.has_globus_compute_gen:
                    result = self._globus_compute_result
                else:
                    result = self._normal_result

                return result(calc_in, Work["persis_info"], self.gen_specs, Work["libE_info"], self.gen_f, Work["tag"])

        else:
            run_gen = []

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}

    def shutdown(self) -> None:
        if self.has_globus_compute_sim:
            self.sim_globus_compute_executor.shutdown()
        if self.has_globus_compute_gen:
            self.gen_globus_compute_executor.shutdown()

    def _get_globus_compute_executor(self):
        try:
            from globus_compute_sdk import Executor
        except ModuleNotFoundError:
            logger.warning("Globus Compute use detected but Globus Compute not importable. Is it installed?")
            logger.warning("Running function evaluations normally on local resources.")
            return None
        else:
            return Executor

    def _truncate_args(self, calc_in, persis_info, specs, libE_info, user_f):
        nparams = len(inspect.signature(user_f).parameters)
        args = [calc_in, persis_info, specs, libE_info]
        return args[:nparams]

    def _normal_result(
        self, calc_in: npt.NDArray, persis_info: dict, specs: dict, libE_info: dict, user_f: Callable, tag: int
    ) -> (npt.NDArray, dict, Optional[int]):
        """User function called in-place"""
        args = self._truncate_args(calc_in, persis_info, specs, libE_info, user_f)
        return user_f(*args)

    def _get_func_uuid(self, tag):
        if tag == EVAL_SIM_TAG:
            return self.globus_compute_simfid
        elif tag == EVAL_GEN_TAG:
            return self.globus_compute_genfid

    def _get_globus_compute_exctr(self, tag):
        if tag == EVAL_SIM_TAG:
            return self.sim_globus_compute_executor
        elif tag == EVAL_GEN_TAG:
            return self.gen_globus_compute_executor

    def _globus_compute_result(
        self, calc_in: npt.NDArray, persis_info: dict, specs: dict, libE_info: dict, user_f: Callable, tag: int
    ) -> (npt.NDArray, dict, Optional[int]):
        """User function submitted to Globus Compute"""
        from libensemble.worker import Worker

        libE_info["comm"] = None  # 'comm' object not pickle-able
        Worker._set_executor(0, None)  # ditto for executor

        fargs = self._truncate_args(calc_in, persis_info, specs, libE_info, user_f)
        exctr = self._get_globus_compute_exctr(tag)
        func_id = self._get_func_uuid(tag)

        task_fut = exctr.submit_to_registered_function(func_id, fargs)
        return task_fut.result()
