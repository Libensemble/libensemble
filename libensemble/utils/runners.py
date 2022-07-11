import logging
import logging.handlers

from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG

logger = logging.getLogger(__name__)


def _funcx_result(funcx_exctr, user_f, calc_in, persis_info, specs, libE_info):
    from libensemble.worker import Worker
    libE_info["comm"] = None  # 'comm' object not pickle-able
    Worker._set_executor(0, None)  # ditto for executor

    future = funcx_exctr.submit(user_f, calc_in, persis_info, specs, libE_info, endpoint_id=specs["funcx_endpoint"])
    remote_exc = future.exception()  # blocks until exception or None
    if remote_exc is None:
        return future.result()
    else:
        raise remote_exc


def _get_funcx_exctr(sim_specs, gen_specs):
    funcx_sim = len(sim_specs.get("funcx_endpoint", "")) > 0
    funcx_gen = len(gen_specs.get("funcx_endpoint", "")) > 0

    if any([funcx_sim, funcx_gen]):
        try:
            from funcx import FuncXClient
            from funcx.sdk.executor import FuncXExecutor

            return FuncXExecutor(FuncXClient()), funcx_sim, funcx_gen
        except ModuleNotFoundError:
            logger.warning("funcX use detected but funcX not importable. Is it installed?")
            return None, False, False
        except Exception:
            return None, False, False
    else:
        return None, False, False


def make_runners(sim_specs, gen_specs):
    """Creates functions to run a sim or gen. These functions are either
    called directly by the worker or submitted to a funcX endpoint."""

    funcx_exctr, funcx_sim, funcx_gen = _get_funcx_exctr(sim_specs, gen_specs)
    sim_f = sim_specs["sim_f"]

    def run_sim(calc_in, persis_info, libE_info):
        """Calls or submits the sim func."""
        if funcx_sim and funcx_exctr:
            return _funcx_result(funcx_exctr, sim_f, calc_in, persis_info, sim_specs, libE_info)
        else:
            return sim_f(calc_in, persis_info, sim_specs, libE_info)

    if gen_specs:
        gen_f = gen_specs["gen_f"]

        def run_gen(calc_in, persis_info, libE_info):
            """Calls or submits the gen func."""
            if funcx_gen and funcx_exctr:
                return _funcx_result(funcx_exctr, gen_f, calc_in, persis_info, gen_specs, libE_info)
            else:
                return gen_f(calc_in, persis_info, gen_specs, libE_info)

    else:
        run_gen = []

    return {EVAL_SIM_TAG: run_sim, EVAL_GEN_TAG: run_gen}
