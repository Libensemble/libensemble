"""
This module is a persistent generation function that performs a uniform
random sample when ``libE_info['persistent']`` isn't ``True``, or performs a
single persistent persistent nlopt local optimization run.
"""

__all__ = ["uniform_or_localopt"]

import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport

import nlopt
import dfols


def uniform_or_localopt(H, persis_info, gen_specs, libE_info):
    """
    This generation function returns ``gen_specs['user']['gen_batch_size']`` uniformly
    sampled points when called in nonpersistent mode (i.e., when
    ``libE_info['persistent']`` isn't ``True``).  Otherwise, the generation
    function starts a persistent nlopt local optimization run.

    .. seealso::
        `test_uniform_sampling_then_persistent_localopt_runs.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_then_persistent_localopt_runs.py>`_ # noqa
    """
    if libE_info.get("persistent"):
        x_opt, persis_info_updates, tag_out = try_and_run_nlopt(H, gen_specs, libE_info)
        # x_opt, persis_info_updates, tag_out = try_and_run_dfols(H, gen_specs, libE_info)
        H_o = []
        return H_o, persis_info_updates, tag_out
    else:
        ub = gen_specs["user"]["ub"]
        lb = gen_specs["user"]["lb"]
        n = len(lb)
        b = gen_specs["user"]["gen_batch_size"]

        H_o = np.zeros(b, dtype=gen_specs["out"])
        for i in range(0, b):
            x = persis_info["rand_stream"].uniform(lb, ub, (1, n))
            H_o = add_to_Out(H_o, x, i, ub, lb)

        persis_info_updates = persis_info  # Send this back so it is overwritten.
        return H_o, persis_info_updates

def try_and_run_dfols(H, gen_specs, libE_info):
    """
    Set up objective and runs nlopt performing communication with the manager in
    order receive function values for points of interest.
    """

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    def dfols_obj_fun(x):

        # Check if we can do an early return
        if np.array_equiv(x, H["x"]):
            return H["fvec"][0]

        # Send back x to the manager, then receive info or stop tag
        H_o = add_to_Out(
            np.zeros(1, dtype=gen_specs["out"]),
            x,
            0,
            gen_specs["user"]["ub"],
            gen_specs["user"]["lb"],
            local=True,
            active=True,
        )
        tag, Work, calc_in = ps.send_recv(H_o)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return np.zeros(100)

        # Return function value (and maybe gradient)
        return calc_in["fvec"][0]

    # ---------------------------------------------------------------------

    x0 = H["x"].flatten()
    lb = gen_specs["user"]["lb"]
    ub = gen_specs["user"]["ub"]
    n = len(ub)

    soln = dfols.solve(dfols_obj_fun, x0, bounds=(lb, ub), rhobeg=0.01)


    x_opt = soln.x

    if soln.flag == soln.EXIT_SUCCESS:
        opt_flag = 1
    else:
        print("[APOSMM] The DFO-LS run started from " + str(x0) + " stopped with an exit "
              "flag of " + str(soln.flag) + ". No point from this run will be "
              "ruled as a minimum! APOSMM may start a new run from some point "
              "in this run.")
        opt_flag = 0

    return x_opt, {}, FINISHED_PERSISTENT_GEN_TAG

def try_and_run_nlopt(H, gen_specs, libE_info):
    """
    Set up objective and runs nlopt performing communication with the manager in
    order receive function values for points of interest.
    """

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    def nlopt_obj_fun(x, grad):

        # Check if we can do an early return
        if np.array_equiv(x, H["x"]):
            if gen_specs["user"]["localopt_method"] in ["LD_MMA"]:
                grad[:] = H["grad"]
            return float(H["f"])

        # Send back x to the manager, then receive info or stop tag
        H_o = add_to_Out(
            np.zeros(1, dtype=gen_specs["out"]),
            x,
            0,
            gen_specs["user"]["ub"],
            gen_specs["user"]["lb"],
            local=True,
            active=True,
        )
        tag, Work, calc_in = ps.send_recv(H_o)
        if tag in [STOP_TAG, PERSIS_STOP]:
            raise nlopt.forced_stop

        # Return function value (and maybe gradient)
        if gen_specs["user"]["localopt_method"] in ["LD_MMA"]:
            grad[:] = calc_in["grad"]
        return float(calc_in["f"])

    # ---------------------------------------------------------------------

    x0 = H["x"].flatten()
    lb = gen_specs["user"]["lb"]
    ub = gen_specs["user"]["ub"]
    n = len(ub)

    opt = nlopt.opt(getattr(nlopt, gen_specs["user"]["localopt_method"]), n)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # Care must be taken with NLopt because a too-large initial step causes
    # nlopt to move the starting point!
    dist_to_bound = min(min(ub - x0), min(x0 - lb))
    init_step = dist_to_bound * gen_specs["user"].get("dist_to_bound_multiple", 1)
    opt.set_initial_step(init_step)

    opt.set_maxeval(gen_specs["user"].get("localopt_maxeval", 100 * n))
    opt.set_min_objective(nlopt_obj_fun)
    opt.set_xtol_rel(gen_specs["user"]["xtol_rel"])

    # Run local optimization.  Only send persis_info_updates back so new
    # information added to persis_info since this persistent instance started
    # (e.g., 'run_order'), is not overwritten
    try:
        x_opt = opt.optimize(x0)
        exit_code = opt.last_optimize_result()
        persis_info_updates = {"done": True}
        if exit_code > 0 and exit_code < 5:
            persis_info_updates["x_opt"] = x_opt
    except Exception:  # Raised when manager sent PERSIS_STOP or STOP_TAG
        x_opt = []
        persis_info_updates = {}

    return x_opt, persis_info_updates, FINISHED_PERSISTENT_GEN_TAG


def add_to_Out(H_o, x, i, ub, lb, local=False, active=False):
    """
    Builds or inserts points into the numpy structured array H_o that will be sent
    back to the manager.
    """
    H_o["x"][i] = x
    H_o["x_on_cube"][i] = (x - lb) / (ub - lb)
    H_o["dist_to_unit_bounds"][i] = np.inf
    H_o["dist_to_better_l"][i] = np.inf
    H_o["dist_to_better_s"][i] = np.inf
    H_o["ind_of_better_l"][i] = -1
    H_o["ind_of_better_s"][i] = -1
    if local:
        H_o["local_pt"] = True
    if active:
        H_o["num_active_runs"] = 1

    return H_o
