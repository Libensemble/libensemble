import logging
import logging.handlers

from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG

logger = logging.getLogger(__name__)


class Runners:
    """Determines and returns methods for workers to run user functions.

    Currently supported: direct-call and funcX
    """

    def __init__(self, sim_specs, gen_specs):
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.sim_f = sim_specs["sim_f"]
        self.gen_f = gen_specs.get("gen_f")
        self.has_funcx_sim = len(sim_specs.get("funcx_endpoint", "")) > 0
        self.has_funcx_gen = len(gen_specs.get("funcx_endpoint", "")) > 0
        self.funcx_exctr = None

        if any([self.has_funcx_sim, self.has_funcx_gen]):
            try:
                from funcx import FuncXClient
                from funcx.sdk.executor import FuncXExecutor

                self.funcx_exctr = FuncXExecutor(FuncXClient())

            except ModuleNotFoundError:
                logger.warning("funcX use detected but funcX not importable. Is it installed?")

    def make_runners(self):
        """Creates functions to run a sim or gen. These functions are either
        called directly by the worker or submitted to a funcX endpoint."""

        def run_sim(calc_in, persis_info, libE_info):
            """Determines how to run sim."""
            if self.has_funcx_sim and self.funcx_exctr:
                result = self._funcx_result
            else:
                result = self._normal_result

            return result(calc_in, persis_info, self.sim_specs, libE_info, self.sim_f)

        if self.gen_specs:

            def run_gen(calc_in, persis_info, libE_info):
                """Determines how to run gen."""
                if self.has_funcx_gen and self.funcx_exctr:
                    result = self._funcx_result
                else:
                    result = self._normal_result

                return result(calc_in, persis_info, self.gen_specs, libE_info, self.gen_f)

        else:
            run_gen = []

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}

    def _normal_result(self, calc_in, persis_info, specs, libE_info, user_f):
        """User function called in-place"""
        return user_f(calc_in, persis_info, specs, libE_info)

    def _funcx_result(self, calc_in, persis_info, specs, libE_info, user_f):
        """User function submitted to funcX"""
        from libensemble.worker import Worker

        libE_info["comm"] = None  # 'comm' object not pickle-able
        Worker._set_executor(0, None)  # ditto for executor

        future = self.funcx_exctr.submit(
            user_f,
            calc_in,
            persis_info,
            specs,
            libE_info,
            endpoint_id=specs["funcx_endpoint"],
        )
        remote_exc = future.exception()  # blocks until exception or None
        if remote_exc is None:
            return future.result()
        else:
            raise remote_exc
