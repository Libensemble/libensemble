import inspect
import logging
import logging.handlers
from typing import Callable, Dict, Optional

import numpy.typing as npt

from libensemble.message_numbers import EVAL_GEN_TAG, EVAL_SIM_TAG

logger = logging.getLogger(__name__)


class Runners:
    """Determines and returns methods for workers to run user functions.

    Currently supported: direct-call and funcX
    """

    def __init__(self, sim_specs: dict, gen_specs: dict) -> None:
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.sim_f = sim_specs["sim_f"]
        self.gen_f = gen_specs.get("gen_f")
        self.has_funcx_sim = len(sim_specs.get("funcx_endpoint", "")) > 0
        self.has_funcx_gen = len(gen_specs.get("funcx_endpoint", "")) > 0

        if any([self.has_funcx_sim, self.has_funcx_gen]):
            if self.has_funcx_sim:
                self.sim_funcx_executor = self._get_funcx_executor()(endpoint_id=self.sim_specs["funcx_endpoint"])
                self.funcx_simfid = self.sim_funcx_executor.register_function(self.sim_f)

            if self.has_funcx_gen:
                self.gen_funcx_executor = self._get_funcx_executor()(endpoint_id=self.gen_specs["funcx_endpoint"])
                self.funcx_genfid = self.gen_funcx_executor.register_function(self.gen_f)

    def make_runners(self) -> Dict[int, Callable]:
        """Creates functions to run a sim or gen. These functions are either
        called directly by the worker or submitted to a funcX endpoint."""

        def run_sim(calc_in, Work):
            """Determines how to run sim."""
            if self.has_funcx_sim:
                result = self._funcx_result
            else:
                result = self._normal_result

            return result(calc_in, Work["persis_info"], self.sim_specs, Work["libE_info"], self.sim_f, Work["tag"])

        if self.gen_specs:

            def run_gen(calc_in, Work):
                """Determines how to run gen."""
                if self.has_funcx_gen:
                    result = self._funcx_result
                else:
                    result = self._normal_result

                return result(calc_in, Work["persis_info"], self.gen_specs, Work["libE_info"], self.gen_f, Work["tag"])

        else:
            run_gen = []

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}

    def shutdown(self) -> None:
        if self.has_funcx_sim:
            self.sim_funcx_executor.shutdown()
        if self.has_funcx_gen:
            self.gen_funcx_executor.shutdown()

    def _get_funcx_executor(self):
        try:
            from funcx import FuncXExecutor
        except ModuleNotFoundError:
            logger.warning("funcX use detected but funcX not importable. Is it installed?")
            logger.warning("Running function evaluations normally on local resources.")
            return None
        else:
            return FuncXExecutor

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
            return self.funcx_simfid
        elif tag == EVAL_GEN_TAG:
            return self.funcx_genfid

    def _get_funcx_exctr(self, tag):
        if tag == EVAL_SIM_TAG:
            return self.sim_funcx_executor
        elif tag == EVAL_GEN_TAG:
            return self.gen_funcx_executor

    def _funcx_result(
        self, calc_in: npt.NDArray, persis_info: dict, specs: dict, libE_info: dict, user_f: Callable, tag: int
    ) -> (npt.NDArray, dict, Optional[int]):
        """User function submitted to funcX"""
        from libensemble.worker import Worker

        libE_info["comm"] = None  # 'comm' object not pickle-able
        Worker._set_executor(0, None)  # ditto for executor

        args = self._truncate_args(calc_in, persis_info, specs, libE_info, user_f)
        exctr = self._get_funcx_exctr(tag)

        task_fut = exctr.submit_to_registered_function(self._get_func_uuid(tag), *args)
        return task_fut.result()
