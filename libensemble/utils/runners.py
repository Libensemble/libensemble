import concurrent
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
            self.funcx_client = self.get_funcx_client()
            self.session_id = self.funcx_client.session_task_group_id
            if self.has_funcx_sim:
                self.funcx_simfid = self.funcx_client.register_function(self.sim_f)
                self.sim_batch_size = self.sim_specs.get("funcx_batch_size")
                self.sim_batch = self.funcx_client.create_batch()

            if self.has_funcx_gen:
                self.funcx_genfid = self.funcx_client.register_function(self.gen_f)
                self.gen_batch_size = self.gen_specs.get("funcx_batch_size")
                self.gen_batch = self.funcx_client.create_batch()

    def make_runners(self) -> Dict[int, Callable]:
        """Creates functions to run a sim or gen. These functions are either
        called directly by the worker or submitted to a funcX endpoint."""

        def run_sim(calc_in, Work):
            """Determines how to run sim."""
            # if self.has_funcx_sim and self.funcx_client:
            #     result = self._funcx_result
            # else:
            result = self._normal_result

            return result(calc_in, Work["persis_info"], self.sim_specs, Work["libE_info"], self.sim_f, Work["tag"])

        if self.gen_specs:

            def run_gen(calc_in, Work):
                """Determines how to run gen."""
                # if self.has_funcx_gen and self.funcx_client:
                #     result = self._funcx_result
                # else:
                result = self._normal_result

                return result(calc_in, Work["persis_info"], self.gen_specs, Work["libE_info"], self.gen_f, Work["tag"])

        else:
            run_gen = []

        return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}

    def get_funcx_client(self):
        try:
            from funcx import FuncXClient
        except ModuleNotFoundError:
            logger.warning("funcX use detected but funcX not importable. Is it installed?")
            logger.warning("Running function evaluations normally on local resources.")
            return None
        else:
            return FuncXClient()

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

    def _batch_result(self, batch, endpoint):
        from funcx import FuncXExecutor

        with FuncXExecutor(endpoint_id=endpoint, task_group_id=self.session_id) as fxe:
            futures = fxe.reload_tasks()
            for f in concurrent.futures.as_completed(futures):
                return f.result()

    def _funcx_result(
        self, calc_in: npt.NDArray, persis_info: dict, specs: dict, libE_info: dict, user_f: Callable, tag: int
    ) -> (npt.NDArray, dict, Optional[int]):
        """User function submitted to funcX"""
        from libensemble.worker import Worker

        libE_info["comm"] = None  # 'comm' object not pickle-able
        Worker._set_executor(0, None)  # ditto for executor

        args = self._truncate_args(calc_in, persis_info, specs, libE_info, user_f)
        if tag == EVAL_SIM_TAG:
            if len(self.sim_batch.tasks) < self.sim_batch_size:
                self.sim_batch.add(self.funcx_simfid, self.sim_specs.get("funcx_endpoint"), args=args)
            else:  # but what if the manager isn't sending any more work, and we haven't hit the batch size limit?
                self.funcx_client.batch_run(self.sim_batch)
                return self._batch_result(self.sim_batch, self.sim_specs.get("funcx_endpoint"))

        elif tag == EVAL_GEN_TAG:
            if len(self.gen_batch.tasks) < self.gen_batch_size:
                self.gen_batch.add(self.funcx_genfid, self.gen_specs.get("funcx_endpoint"), args=args)
            else:
                self.funcx_client.batch_run(self.gen_batch)
                return self._batch_result(self.gen_batch, self.gen_specs.get("funcx_endpoint"))

        # TODO: But what *can* I return, if anything, to signal that the worker should still be sent work?
        # Is there a fundamental problem with this architecture? Should this worker be "pretending" to be
        # persistent so that we can send batches of work to it?

        # future = self.funcx_exctr.submit(
        #     user_f,
        #     calc_in,
        #     persis_info,
        #     specs,
        #     libE_info,
        #     endpoint_id=specs["funcx_endpoint"],
        # )
        # remote_exc = future.exception()  # blocks until exception or None
        # if remote_exc is None:
        #     return future.result()
        # else:
        #     raise remote_exc
