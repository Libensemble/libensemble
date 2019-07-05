__all__ = ['uniform_or_localopt']

import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.gen_funcs.support import sendrecv_mgr_worker_msg

import nlopt


def uniform_or_localopt(H, persis_info, gen_specs, libE_info):
    """
    This generation function returns ``gen_specs['gen_batch_size']`` uniformly
    sampled points when called in nonpersistent mode (i.e., when
    ``libE_info['persistent']`` isn't ``True``).  Otherwise, the generation
    function a persistent nlopt local optimization run.

    :See:
        ``libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling_with_persistent_localopt_gens.py``
    """

    if libE_info.get('persistent'):
        x_opt, persis_info_updates, tag_out = try_and_run_nlopt(H, gen_specs, libE_info)
        O = []
        return O, persis_info_updates, tag_out
    else:
        ub = gen_specs['ub']
        lb = gen_specs['lb']
        n = len(lb)
        b = gen_specs['gen_batch_size']

        O = np.zeros(b, dtype=gen_specs['out'])
        for i in range(0, b):
            x = persis_info['rand_stream'].uniform(lb, ub, (1, n))
            O = add_to_O(O, x, i, ub, lb)

        persis_info_updates = persis_info  # Send this back so it is overwritten.
        return O, persis_info_updates


def try_and_run_nlopt(H, gen_specs, libE_info):
    """
    Set up objective and runs nlopt performing communication with the manager in
    order receive function values for points of interest.
    """

    comm = libE_info['comm']

    def nlopt_obj_fun(x, grad):

        # Check if we can do an early return
        if np.array_equiv(x, H['x']):
            if gen_specs['localopt_method'] in ['LD_MMA']:
                grad[:] = H['grad']
            return np.float(H['f'])

        # Send back x to the manager, then receive info or stop tag
        O = add_to_O(np.zeros(1, dtype=gen_specs['out']), x, 0,
                     gen_specs['ub'], gen_specs['lb'], local=True, active=True)
        tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, O)
        if tag in [STOP_TAG, PERSIS_STOP]:
            nlopt.forced_stop.message = 'tag=' + str(tag)
            raise nlopt.forced_stop

        # Return function value (and maybe gradient)
        if gen_specs['localopt_method'] in ['LD_MMA']:
            grad[:] = calc_in['grad']
        return float(calc_in['f'])

    # ---------------------------------------------------------------------

    x0 = H['x'].flatten()
    lb = gen_specs['lb']
    ub = gen_specs['ub']
    n = len(ub)

    opt = nlopt.opt(getattr(nlopt, gen_specs['localopt_method']), n)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # Care must be taken with NLopt because a too-large initial step causes
    # nlopt to move the starting point!
    dist_to_bound = min(min(ub-x0), min(x0-lb))
    init_step = dist_to_bound*gen_specs.get('dist_to_bound_multiple', 1)
    opt.set_initial_step(init_step)

    opt.set_maxeval(gen_specs.get('localopt_maxeval', 100*n))
    opt.set_min_objective(nlopt_obj_fun)
    opt.set_xtol_rel(gen_specs['xtol_rel'])

    # Run local optimization.  Only send persis_info_updates back so new
    # information added to persis_info since this persistent instance started
    # (e.g., 'run_order'), is not overwritten
    try:
        x_opt = opt.optimize(x0)
        exit_code = opt.last_optimize_result()
        persis_info_updates = {'done': True}
        if exit_code > 0 and exit_code < 5:
            persis_info_updates['x_opt'] = x_opt
        tag_out = FINISHED_PERSISTENT_GEN_TAG
    except Exception as e:  # Raised when manager sent PERSIS_STOP or STOP_TAG
        x_opt = []
        persis_info_updates = {}
        tag_out = int(e.message.split('=')[-1])

    return x_opt, persis_info_updates, tag_out


def add_to_O(O, x, i, ub, lb, local=False, active=False):
    """
    Builds or inserts points into the numpy structured array O that will be sent
    back to the manager.
    """
    O['x'][i] = x
    O['x_on_cube'][i] = (x-lb)/(ub-lb)
    O['dist_to_unit_bounds'][i] = np.inf
    O['dist_to_better_l'][i] = np.inf
    O['dist_to_better_s'][i] = np.inf
    O['ind_of_better_l'][i] = -1
    O['ind_of_better_s'][i] = -1
    if local:
        O['local_pt'] = True
    if active:
        O['num_active_runs'] = 1

    return O
