"""
This module contains methods used our implementation of the Asynchronously
Parallel Optimization Solver for finding Multiple Minima (APOSMM) method
described in detail in the paper
`https://doi.org/10.1007/s12532-017-0131-4 <https://doi.org/10.1007/s12532-017-0131-4>`_

This implementation of APOSMM was developed by Kaushik Kulkarni and Jeffrey
Larson in the summer of 2019.
"""
__all__ = ['initialize_APOSMM', 'decide_where_to_start_localopt', 'update_history_dist']

import numpy as np
from scipy.spatial.distance import cdist
from scipy import optimize as sp_opt
from petsc4py import PETSc

from mpi4py import MPI

from math import log, gamma, pi, sqrt

import nlopt
import dfols

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP
from libensemble.tools.gen_support import send_mgr_worker_msg
from libensemble.tools.gen_support import get_mgr_worker_msg


from multiprocessing import Event, Process, Queue


class APOSMMException(Exception):
    "Raised for any exception in APOSMM"


class ConvergedMsg(object):
    """
    Message communicated when a local optimization is converged.
    """
    def __init__(self, x):
        self.x = x


class ErrorMsg(object):
    """
    Message communicated when a local optimization has an exception.
    """
    def __init__(self, x):
        self.x = x


def aposmm(H, persis_info, gen_specs, libE_info):
    """
    APOSMM coordinates multiple local optimization runs, starting from points
    which do not have a better point nearby (within a distance ``r_k``). This
    generation function produces/requires the following fields in ``H``:

    - ``'x' [n floats]``: Parameters being optimized over
    - ``'x_on_cube' [n floats]``: Parameters scaled to the unit cube
    - ``'f' [float]``: Objective function being minimized
    - ``'local_pt' [bool]``: True if point from a local optimization run
    - ``'dist_to_unit_bounds' [float]``: Distance to domain boundary
    - ``'dist_to_better_l' [float]``: Dist to closest better local opt point
    - ``'dist_to_better_s' [float]``: Dist to closest better sample point
    - ``'ind_of_better_l' [int]``: Index of point ``'dist_to_better_l``' away
    - ``'ind_of_better_s' [int]``: Index of point ``'dist_to_better_s``' away
    - ``'started_run' [bool]``: True if point has started a local opt run
    - ``'num_active_runs' [int]``: Number of active local runs point is in
    - ``'local_min' [float]``: True if point has been ruled a local minima
    - ``'sim_id' [int]``: Row number of entry in history

    and optionally

    - ``'fvec' [m floats]``: All objective components (if calculated together)
    - ``'obj_component' [int]``: Index corresponding to value in ``'f_i``'
    - ``'pt_id' [int]``: Identify the point (useful when evaluating different
      objective components for a given ``'x'``)

    When using libEnsemble to do individual objective component evaluations,
    APOSMM will return ``gen_specs['user']['components']`` copies of each point, but
    the component=0 entry of each point will only be considered when

    - deciding where to start a run,
    - best nearby point,
    - storing the order of the points is the run
    - storing the combined objective function value
    - etc

    Necessary quantities in ``gen_specs['user']`` are:

    - ``'lb' [n floats]``: Lower bound on search domain
    - ``'ub' [n floats]``: Upper bound on search domain
    - ``'initial_sample_size' [int]``: Number of uniformly sampled points
      must be returned (non-nan value) before a local opt run is started

    - ``'localopt_method' [str]``: Name of an NLopt, PETSc/TAO, or SciPy method
      (see 'advance_local_run' below for supported methods)

    Optional ``gen_specs['user']`` entries are:

    - ``'sample_points' [numpy array]``: Points to be sampled (original domain)
    - ``'combine_component_func' [func]``: Function to combine obj components
    - ``'components' [int]``: Number of objective components
    - ``'dist_to_bound_multiple' [float in (0,1]]``: What fraction of the
      distance to the nearest boundary should the initial step size be in
      localopt runs
    - ``'lhs_divisions' [int]``: Number of Latin hypercube sampling partitions
      (0 or 1 results in uniform sampling)
    - ``'mu' [float]``: Distance from the boundary that all localopt starting
      points must satisfy
    - ``'nu' [float]``: Distance from identified minima that all starting
      points must satisfy
    - ``'rk_const' [float]``: Multiplier in front of the r_k value
    - ``'max_active_runs' [int]``: Bound on number of runs APOSMM is advancing

    And ``gen_specs['user']`` convergence tolerances for NLopt, PETSc/TAO, SciPy

    - ``'fatol' [float]``:
    - ``'ftol_abs' [float]``:
    - ``'ftol_rel' [float]``:
    - ``'gatol' [float]``:
    - ``'grtol' [float]``:
    - ``'xtol_abs' [float]``:
    - ``'xtol_rel' [float]``:
    - ``'tol' [float]``:


    As a default, APOSMM starts a local optimization runs from a point that:

    - is not in an active local optimization run,
    - is more than ``mu`` from the boundary (in the unit-cube domain),
    - is more than ``nu`` from identified minima (in the unit-cube domain),
    - does not have a better point within a distance ``r_k`` of it.

    If the above results in more than ``'max_active_runs'`` being advanced, the
    best point in each run is determined and the dist_to_better is computed
    (with inf being the value for the best run). Then those
    ``'max_active_runs'`` runs with largest dist_to_better are advanced
    (breaking ties arbitrarily).

    :Note:
        ``gen_specs['user']['combine_component_func']`` must be defined when there are
        multiple objective components.

    :Note:
        APOSMM critically uses ``persis_info`` to store information about
        active runs, order of points in each run, etc. The allocation function
        must ensure it's always given.

    .. seealso::
        `test_branin_aposmm_nlopt_and_then_scipy.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_branin_aposmm_nlopt_and_then_scipy.py>`_
        for basic APOSMM usage.

    .. seealso::
        `test_chwirut_aposmm_one_residual_at_a_time.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_chwirut_aposmm_one_residual_at_a_time.py>`_
        for an example of APOSMM coordinating multiple local optimization runs
        for an objective with more than one component.
    """
    """
    Description of intermediate variables in aposmm_logic:

    n:                domain dimension
    n_s:              the number of complete evaluations of sampled points
    updated_inds:     indices of H that have been updated (and so all their
                      information must be sent back to libE manager to update)
    O:                new points to be sent back to the history


    When re-running a local opt method to get the next point:
    advance_local_run.x_new:      stores the first new point requested by
                                  a local optimization method
    advance_local_run.pt_in_run:  counts function evaluations to know
                                  when a new point is given

    starting_inds:    indices where a runs should be started.
    active_runs:      indices of active local optimization runs
    sorted_run_inds:  indices of the considered run (in the order they were
                      requested by the localopt method)
    x_opt:            the reported minimum from a localopt run (disregarded
                      unless exit_code isn't 0)
    exit_code:        0 if a new localopt point has been found, otherwise it's
                      the NLopt/TAO/SciPy code
    samples_needed:   Number of additional uniformly drawn samples needed


    Description of persistent variables used to maintain the state of APOSMM

    persis_info['total_runs']: Running count of started/completed localopt runs
    persis_info['run_order']: Sequence of indices of points in unfinished runs
    persis_info['old_runs']: Sequence of indices of points in finished runs

    """
    user_specs = gen_specs['user']

    n, n_s, rk_const, ld, mu, nu, comm, local_H = initialize_APOSMM(H, user_specs, libE_info)
    local_opters, sim_id_to_child_inds, run_order, run_pts, total_runs, fields_to_pass = initialize_children(user_specs)

    if user_specs['initial_sample_size'] != 0:
        # Send our initial sample. We don't need to check that n_s is large enough:
        # the alloc_func only returns when the initial sample has function values.
        persis_info = add_k_sample_points_to_local_H(user_specs['initial_sample_size'], user_specs,
                                                     persis_info, n, comm, local_H,
                                                     sim_id_to_child_inds)
        send_mgr_worker_msg(comm, local_H[-user_specs['initial_sample_size']:][[i[0] for i in gen_specs['out']]])
        something_sent = True
    else:
        something_sent = False

    calc_status = None
    tag = None
    first_pass = True
    while 1:
        new_opt_inds_to_send_mgr = []
        new_inds_to_send_mgr = []

        if something_sent:
            tag, Work, calc_in = get_mgr_worker_msg(comm)

            if tag in [STOP_TAG, PERSIS_STOP]:
                clean_up_and_stop(local_H, local_opters, run_order)
                persis_info['run_order'] = run_order
                calc_status = Work
                break

            n_s, n_r = update_local_H_after_receiving(local_H, n, n_s, user_specs, Work, calc_in, fields_to_pass)

            for row in calc_in:
                if sim_id_to_child_inds.get(row['sim_id']):
                    # Point came from a child local opt run
                    for child_idx in sim_id_to_child_inds[row['sim_id']]:
                        x_new = local_opters[child_idx].iterate(row[fields_to_pass])
                        if isinstance(x_new, ConvergedMsg):
                            x_opt = x_new.x
                            opt_ind = update_history_optimal(x_opt, local_H, run_order[child_idx])
                            new_opt_inds_to_send_mgr.append(opt_ind)
                            local_opters.pop(child_idx)
                        else:
                            add_to_local_H(local_H, x_new, user_specs, local_flag=1, on_cube=True)
                            new_inds_to_send_mgr.append(len(local_H)-1)

                            run_order[child_idx].append(local_H[-1]['sim_id'])
                            run_pts[child_idx].append(x_new)
                            if local_H[-1]['sim_id'] in sim_id_to_child_inds:
                                sim_id_to_child_inds[local_H[-1]['sim_id']] += (child_idx, )
                            else:
                                sim_id_to_child_inds[local_H[-1]['sim_id']] = (child_idx, )

        starting_inds = decide_where_to_start_localopt(local_H, n, n_s, rk_const, ld, mu, nu)

        for ind in starting_inds:
            if len([p for p in local_opters.values() if p.is_running]) < user_specs.get('max_active_runs', np.inf):
                local_H['started_run'][ind] = 1

                # Initialize a local opt run
                local_opter = LocalOptInterfacer(user_specs, local_H[ind]['x_on_cube'],
                                                 local_H[ind]['f'] if 'f' in fields_to_pass else local_H[ind]['fvec'],
                                                 local_H[ind]['grad'] if 'grad' in fields_to_pass else None)

                local_opters[total_runs] = local_opter

                x_new = local_opter.iterate(local_H[ind][fields_to_pass])  # Assuming the second point can't be ruled optimal

                add_to_local_H(local_H, x_new, user_specs, local_flag=1, on_cube=True)
                new_inds_to_send_mgr.append(len(local_H)-1)

                run_order[total_runs] = [ind, local_H[-1]['sim_id']]
                run_pts[total_runs] = [local_H['x_on_cube'], x_new]

                if local_H[-1]['sim_id'] in sim_id_to_child_inds:
                    sim_id_to_child_inds[local_H[-1]['sim_id']] += (total_runs, )
                else:
                    sim_id_to_child_inds[local_H[-1]['sim_id']] = (total_runs, )

                total_runs += 1

        if first_pass:
            num_samples_needed = user_specs.get('num_pts_first_pass', 1) - len(new_inds_to_send_mgr)
            first_pass = False
        else:
            num_samples_needed = n_r-len(new_inds_to_send_mgr)

        if num_samples_needed > 0:
            persis_info = add_k_sample_points_to_local_H(num_samples_needed, user_specs, persis_info, n, comm, local_H, sim_id_to_child_inds)
            new_inds_to_send_mgr = new_inds_to_send_mgr + list(range(len(local_H)-num_samples_needed, len(local_H)))

        send_mgr_worker_msg(comm, local_H[new_inds_to_send_mgr + new_opt_inds_to_send_mgr][[i[0] for i in gen_specs['out']]])
        something_sent = True

    return [], persis_info, calc_status


class LocalOptInterfacer(object):
    def __init__(self, user_specs, x0, f0, grad0=None):
        """
        :param x0: A numpy array of the initial guess solution. This guess
            should be scaled to a unit cube.
        :param f0: A numpy array of the initial function value.


        .. warning:: In order to have correct functioning of the local
            optimization child processes. ~self.iterate~ should be called
            immediately after creating the class.

        """
        self.parent_can_read = Event()

        self.comm_queue = Queue()
        self.child_can_read = Event()

        self.x0 = x0.copy()
        self.f0 = f0.copy()
        if grad0 is not None:
            self.grad0 = grad0.copy()
        else:
            self.grad0 = None

        # Setting the local optimization method
        if user_specs['localopt_method'] in ['LN_SBPLX', 'LN_BOBYQA', 'LN_COBYLA', 'LN_NEWUOA', 'LN_NELDERMEAD', 'LD_MMA']:
            run_local_opt = run_local_nlopt
        elif user_specs['localopt_method'] in ['pounders', 'blmvm', 'nm']:
            run_local_opt = run_local_tao
        elif user_specs['localopt_method'] in ['scipy_Nelder-Mead', 'scipy_COBYLA']:
            run_local_opt = run_local_scipy_opt
        elif user_specs['localopt_method'] in ['dfols']:
            run_local_opt = run_local_dfols
        elif user_specs['localopt_method'] in ['external_localopt']:
            run_local_opt = run_external_localopt

        self.parent_can_read.clear()
        self.process = Process(target=opt_runner, args=(run_local_opt, user_specs,
                               self.comm_queue, x0, f0, self.child_can_read,
                               self.parent_can_read))

        self.process.start()
        self.is_running = True
        self.parent_can_read.wait()
        x_new = self.comm_queue.get()
        if isinstance(x_new, ErrorMsg):
            raise APOSMMException(x_new.x)

        assert np.allclose(x_new, x0, rtol=1e-15, atol=1e-15), "The first point requested by this run does not match the starting point. Exiting"

    def iterate(self, data):
        """
        Returns an instance of either :class:`numpy.ndarray` corresponding to the next
        iterative guess or :class:`ConvergedMsg` when the solver is converged.

        :param f: A numpy array of the function evaluation.
        :param grad: A numpy array of the function's gradient.
        """
        self.parent_can_read.clear()

        if 'grad' in data.dtype.names:
            self.comm_queue.put((data['x_on_cube'], data['f'], data['grad']))
        elif 'fvec' in data.dtype.names:
            self.comm_queue.put((data['x_on_cube'], data['fvec']))
        else:
            self.comm_queue.put((data['x_on_cube'], data['f'], ))

        self.child_can_read.set()
        self.parent_can_read.wait()

        x_new = self.comm_queue.get()
        if isinstance(x_new, ErrorMsg):
            raise APOSMMException(x_new.x)
        elif isinstance(x_new, ConvergedMsg):
            self.process.join()
            self.comm_queue.close()
            self.comm_queue.join_thread()
            self.is_running = False
        else:
            x_new = np.atleast_2d(x_new)

        return x_new

    def destroy(self, previous_x):

        while not isinstance(previous_x, ConvergedMsg):
            self.parent_can_read.clear()
            if self.grad0 is None:
                self.comm_queue.put((previous_x, 0*np.ones_like(self.f0),))
            else:
                self.comm_queue.put((previous_x, 0*np.ones_like(self.f0), np.zeros_like(self.grad0)))

            self.child_can_read.set()
            self.parent_can_read.wait()

            previous_x = self.comm_queue.get()
        assert isinstance(previous_x, ConvergedMsg)
        self.process.join()
        self.comm_queue.close()
        self.comm_queue.join_thread()
        self.is_running = False


def run_local_nlopt(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    # print('[Child]: Started local opt at {}.'.format(x0), flush=True)
    n = len(user_specs['ub'])

    opt = nlopt.opt(getattr(nlopt, user_specs['localopt_method']), n)

    lb = np.zeros(n)
    ub = np.ones(n)

    if not user_specs.get('periodic'):
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

    # Care must be taken here because a too-large initial step causes nlopt to move the starting point!
    dist_to_bound = min(min(ub-x0), min(x0-lb))
    assert dist_to_bound > np.finfo(np.float32).eps, "The distance to the boundary is too small for NLopt to handle"

    if 'dist_to_bound_multiple' in user_specs:
        opt.set_initial_step(dist_to_bound*user_specs['dist_to_bound_multiple'])
    else:
        opt.set_initial_step(dist_to_bound)

    opt.set_maxeval(user_specs.get('run_max_eval', 1000*n))

    opt.set_min_objective(lambda x, grad: nlopt_callback_fun(x, grad,
                          comm_queue, child_can_read, parent_can_read,
                          user_specs))

    if 'xtol_rel' in user_specs:
        opt.set_xtol_rel(user_specs['xtol_rel'])
    if 'ftol_rel' in user_specs:
        opt.set_ftol_rel(user_specs['ftol_rel'])
    if 'xtol_abs' in user_specs:
        opt.set_xtol_abs(user_specs['xtol_abs'])
    if 'ftol_abs' in user_specs:
        opt.set_ftol_abs(user_specs['ftol_abs'])

    # FIXME: Do we need to do something of the final 'x_opt'?
    # print('[Child]: Started my optimization', flush=True)
    x_opt = opt.optimize(x0)

    if user_specs.get('periodic'):
        x_opt = x_opt % 1  # Shift x_opt to be in the correct location in the unit cube (not the domain user_specs['lb'] - user_specs['ub'])

    finish_queue(x_opt, comm_queue, parent_can_read, user_specs)


def run_local_scipy_opt(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):

    # Construct the bounds in the form of constraints
    cons = []
    for factor in range(len(x0)):
        lo = {'type': 'ineq',
              'fun': lambda x, lb=user_specs['lb'][factor], i=factor: x[i]-lb}
        up = {'type': 'ineq',
              'fun': lambda x, ub=user_specs['ub'][factor], i=factor: ub-x[i]}
        cons.append(lo)
        cons.append(up)

    method = user_specs['localopt_method'][6:]
    # print('[Child]: Started my optimization', flush=True)
    res = sp_opt.minimize(lambda x: scipy_dfols_callback_fun(x, comm_queue,
                          child_can_read, parent_can_read, user_specs), x0,
                          # constraints=cons,
                          method=method, **user_specs.get('scipy_kwargs', {}))

    # if res['status'] == 2:  # SciPy code for exhausting budget of evaluations, so not at a minimum
    #     exit_code = 0
    # else:
    #     if method == 'Nelder-Mead':
    #         assert res['status'] == 0, "Unknown status for Nelder-Mead"
    #         exit_code = 1

    if user_specs.get('periodic'):
        x_opt = res['x'] % 1
    else:
        x_opt = res['x']

    # FIXME: Need to do something with the exit codes.
    # print(exit_code)

    finish_queue(x_opt, comm_queue, parent_can_read, user_specs)


def run_external_localopt(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):

    import subprocess
    import os
    from uuid import uuid4

    run_id = uuid4().hex

    x_file = 'x_' + run_id + '.txt'
    y_file = 'y_' + run_id + '.txt'
    x_done_file = 'x_done_' + run_id + '.txt'
    y_done_file = 'y_done_' + run_id + '.txt'
    opt_file = 'opt_' + run_id + '.txt'

    # cmd = ["matlab", "-nodisplay", "-nodesktop", "-nojvm", "-nosplash", "-r",
    cmd = ["octave", "--no-window-system", "--eval",
           "x0=[" + " ".join(["{:18.18f}".format(x) for x in x0]) + "];"
           "opt_file='" + opt_file + "';"
           "x_file='" + x_file + "';"
           "y_file='" + y_file + "';"
           "x_done_file='" + x_done_file + "';"
           "y_done_file='" + y_done_file + "';"
           "call_matlab_octave_script"]

    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.DEVNULL)

    while p.poll() is None:  # Process still going
        if os.path.isfile(x_done_file):  # x file exists
            x = np.loadtxt(x_file)
            os.remove(x_done_file)

            x_recv, f_recv = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)

            np.savetxt(y_file, [f_recv])
            open(y_done_file, 'w').close()

    x_opt = np.loadtxt(opt_file)

    for f in [x_file, y_file, opt_file]:
        os.remove(f)

    finish_queue(x_opt, comm_queue, parent_can_read, user_specs)


def run_local_dfols(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):

    # Define bound constraints (lower <= x <= upper)
    lb = np.zeros(len(x0))
    ub = np.ones(len(x0))

    # Set random seed (for reproducibility)
    np.random.seed(0)

    # Care must be taken here because a too-large initial step causes DFO-LS to move the starting point!
    dist_to_bound = min(min(ub-x0), min(x0-lb))
    assert dist_to_bound > np.finfo(np.float32).eps, "The distance to the boundary is too small"
    assert 'bounds' not in user_specs.get('dfols_kwargs', {}), "APOSMM must set the bounds for DFO-LS"
    assert 'rhobeg' not in user_specs.get('dfols_kwargs', {}), "APOSMM must set rhobeg for DFO-LS"
    assert 'x0' not in user_specs.get('dfols_kwargs', {}), "APOSMM must set x0 for DFO-LS"

    # Call DFO-LS
    soln = dfols.solve(lambda x: scipy_dfols_callback_fun(x, comm_queue, child_can_read, parent_can_read, user_specs),
                       x0, bounds=(lb, ub), rhobeg=0.5*dist_to_bound, **user_specs.get('dfols_kwargs', {}))

    x_opt = soln.x

    # FIXME: Need to do something with the exit codes.
    # print(exit_code)

    finish_queue(x_opt, comm_queue, parent_can_read, user_specs)


def run_local_tao(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):

    assert isinstance(x0, np.ndarray)

    tao_comm = MPI.COMM_SELF
    n, = x0.shape
    if f0.shape == ():
        m = 1
    else:
        m, = f0.shape

    # Create starting point, bounds, and tao object
    x = PETSc.Vec().create(tao_comm)
    x.setSizes(n)
    x.setFromOptions()
    x.array = x0
    lb = x.duplicate()
    ub = x.duplicate()
    lb.array = 0*np.ones(n)
    ub.array = 1*np.ones(n)
    tao = PETSc.TAO().create(tao_comm)
    tao.setType(user_specs['localopt_method'])

    if user_specs['localopt_method'] == 'pounders':
        f = PETSc.Vec().create(tao_comm)
        f.setSizes(m)
        f.setFromOptions()

        if hasattr(tao, 'setResidual'):
            tao.setResidual(lambda tao, x, f: tao_callback_fun_pounders(tao, x, f, comm_queue, child_can_read, parent_can_read, user_specs), f)
        else:
            tao.setSeparableObjective(lambda tao, x, f: tao_callback_fun_pounders(tao, x, f, comm_queue, child_can_read, parent_can_read, user_specs), f)
        delta_0 = user_specs['dist_to_bound_multiple']*np.min([np.min(ub.array-x.array), np.min(x.array-lb.array)])
        PETSc.Options().setValue('-tao_pounders_delta', str(delta_0))

    elif user_specs['localopt_method'] == 'nm':
        tao.setObjective(lambda tao, x: tao_callback_fun_nm(tao, x, comm_queue, child_can_read, parent_can_read, user_specs))

    elif user_specs['localopt_method'] == 'blmvm':
        g = PETSc.Vec().create(tao_comm)
        g.setSizes(n)
        g.setFromOptions()
        tao.setObjectiveGradient(lambda tao, x, g: tao_callback_fun_grad(tao, x, g, comm_queue, child_can_read, parent_can_read, user_specs))

    # Set everything for tao before solving
    # FIXME: Hard-coding 100 as the max funcs as couldn't find any other
    # sensible value.
    PETSc.Options().setValue('-tao_max_funcs', str(user_specs.get('run_max_eval', 1000*n)))
    tao.setFromOptions()
    tao.setVariableBounds((lb, ub))

    tao.setTolerances(grtol=user_specs.get('grtol', 1e-8), gatol=user_specs.get('gatol', 1e-8))
    tao.setInitial(x)

    # print('[Child]: Started my optimization', flush=True)
    tao.solve(x)

    x_opt = tao.getSolution().getArray()
    # exit_code = tao.getConvergedReason()

    # FIXME: Need to do something with the exit codes.
    # print(exit_code)
    # print(tao.view())
    # print(x_opt)

    if user_specs['localopt_method'] == 'pounders':
        f.destroy()
    elif user_specs['localopt_method'] == 'blmvm':
        g.destroy()

    lb.destroy()
    ub.destroy()
    x.destroy()
    tao.destroy()

    finish_queue(x_opt, comm_queue, parent_can_read, user_specs)


def opt_runner(run_local_opt, user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    try:
        run_local_opt(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read)
    except Exception as e:
        comm_queue.put(ErrorMsg(e))
        parent_can_read.set()


# Callback functions and routines
def nlopt_callback_fun(x, grad, comm_queue, child_can_read, parent_can_read, user_specs):

    if user_specs['localopt_method'] in ['LD_MMA']:
        x_recv, f_recv, grad_recv = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)
        grad[:] = grad_recv
    else:
        assert user_specs['localopt_method'] in ['LN_SBPLX', 'LN_BOBYQA', 'LN_NEWUOA',
                                                 'LN_COBYLA', 'LN_NELDERMEAD', 'LD_MMA']
        x_recv, f_recv = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)

    return f_recv


def scipy_dfols_callback_fun(x, comm_queue, child_can_read, parent_can_read, user_specs):

    x_recv, f_x_recv, = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)

    return f_x_recv


def tao_callback_fun_nm(tao, x, comm_queue, child_can_read, parent_can_read, user_specs):

    x_recv, f_recv, = put_set_wait_get(x.array_r, comm_queue, parent_can_read, child_can_read, user_specs)

    return f_recv


def tao_callback_fun_pounders(tao, x, f, comm_queue, child_can_read, parent_can_read, user_specs):

    x_recv, f_recv, = put_set_wait_get(x.array_r, comm_queue, parent_can_read, child_can_read, user_specs)
    f.array[:] = f_recv

    return f


def tao_callback_fun_grad(tao, x, g, comm_queue, child_can_read, parent_can_read, user_specs):

    x_recv, f_recv, grad_recv = put_set_wait_get(x.array_r, comm_queue, parent_can_read, child_can_read, user_specs)
    g.array[:] = grad_recv

    return f_recv


def finish_queue(x_opt, comm_queue, parent_can_read, user_specs):

    if user_specs.get('print'):
        print('Local optimum on the [0,1]^n domain', x_opt, flush=True)
    comm_queue.put(ConvergedMsg(x_opt))
    parent_can_read.set()


def put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs):
    """This routine is used by children process callback functions. It:
    - puts x into a comm_queue,
    - tells the parent it can read,
    - tells the child to wait
    - receives the values put in the comm_queue by the parent
    - checks that the first value received matches x
    - removes the wait on the child
    - returns values"""

    comm_queue.put(x)
    # print('[Child]: I just put x_on_cube =', x.array, flush=True)
    # print('[Child]: Parent should no longer wait.', flush=True)
    parent_can_read.set()
    # print('[Child]: I have started waiting', flush=True)
    child_can_read.wait()
    # print('[Child]: Wohooo.. I am free folks', flush=True)
    values = comm_queue.get()
    child_can_read.clear()

    if user_specs.get('periodic'):
        assert np.allclose(x % 1, values[0] % 1, rtol=1e-15, atol=1e-15), "The point I gave is not the point I got back!"
    else:
        assert np.allclose(x, values[0], rtol=1e-15, atol=1e-15), "The point I gave is not the point I got back!"

    return values


def update_local_H_after_receiving(local_H, n, n_s, user_specs, Work, calc_in, fields_to_pass):

    for name in ['f', 'x_on_cube', 'grad', 'fvec']:
        if name in fields_to_pass:
            assert name in calc_in.dtype.names, name + " must be returned to persistent_aposmm for localopt_method" + user_specs['localopt_method']

    for name in calc_in.dtype.names:
        local_H[name][Work['libE_info']['H_rows']] = calc_in[name]

    local_H['returned'][Work['libE_info']['H_rows']] = True
    n_s += np.sum(~local_H[Work['libE_info']['H_rows']]['local_pt'])
    n_r = len(Work['libE_info']['H_rows'])

    # dist -> distance
    update_history_dist(local_H, n)

    return n_s, n_r


def add_to_local_H(local_H, pts, user_specs, local_flag=0, sorted_run_inds=[], run=[], on_cube=True):
    """
    Adds points to O, the numpy structured array to be sent back to the manager
    """
    assert not local_flag or len(pts) == 1, "Can't > 1 local points"

    len_local_H = len(local_H)

    ub = user_specs['ub']
    lb = user_specs['lb']

    num_pts = len(pts)

    local_H.resize(len(local_H)+num_pts, refcheck=False)  # Adds num_pts rows of zeros to O

    if on_cube:
        local_H['x_on_cube'][-num_pts:] = pts
        local_H['x'][-num_pts:] = pts*(ub-lb)+lb
    else:
        local_H['x_on_cube'][-num_pts:] = (pts-lb)/(ub-lb)
        local_H['x'][-num_pts:] = pts

    if user_specs.get('periodic'):
        local_H['x_on_cube'][-num_pts:] = local_H['x_on_cube'][-num_pts:] % 1

    local_H['sim_id'][-num_pts:] = np.arange(len_local_H, len_local_H+num_pts)
    local_H['local_pt'][-num_pts:] = local_flag

    initialize_dists_and_inds(local_H, num_pts)

    if local_flag:
        local_H['num_active_runs'][-num_pts] += 1


def initialize_dists_and_inds(local_H, num_pts):
    local_H['dist_to_unit_bounds'][-num_pts:] = np.inf
    local_H['dist_to_better_l'][-num_pts:] = np.inf
    local_H['dist_to_better_s'][-num_pts:] = np.inf
    local_H['ind_of_better_l'][-num_pts:] = -1
    local_H['ind_of_better_s'][-num_pts:] = -1


def update_history_dist(H, n):
    """
    Updates distances/indices after new points that have been evaluated.

    .. seealso::
        `start_persistent_local_opt_gens.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_persistent_local_opt_gens.py>`_
    """

    new_inds = np.where(~H['known_to_aposmm'])[0]

    p = np.logical_and.reduce((H['returned'], ~np.isnan(H['f'])))

    for new_ind in new_inds:
        # Loop over new returned points and update their distances
        if p[new_ind]:
            H['known_to_aposmm'][new_ind] = True

            # Compute distance to boundary
            H['dist_to_unit_bounds'][new_ind] = min(min(np.ones(n)-H['x_on_cube'][new_ind]), min(H['x_on_cube'][new_ind]-np.zeros(n)))

            dist_to_all = cdist(H['x_on_cube'][[new_ind]], H['x_on_cube'][p], 'euclidean').flatten()
            new_better_than = H['f'][new_ind] < H['f'][p]

            # Update any other points if new_ind is closer and better
            if H['local_pt'][new_ind]:
                inds_of_p = np.logical_and(dist_to_all < H['dist_to_better_l'][p], new_better_than)
                updates = np.where(p)[0][inds_of_p]
                H['dist_to_better_l'][updates] = dist_to_all[inds_of_p]
                H['ind_of_better_l'][updates] = new_ind
            else:
                inds_of_p = np.logical_and(dist_to_all < H['dist_to_better_s'][p], new_better_than)
                updates = np.where(p)[0][inds_of_p]
                H['dist_to_better_s'][updates] = dist_to_all[inds_of_p]
                H['ind_of_better_s'][updates] = new_ind

            # Since we allow equality when deciding better_than_new_l and
            # better_than_new_s, we have to prevent new_ind from being its own
            # better point.
            better_than_new_l = np.logical_and.reduce((~new_better_than, H['local_pt'][p], H['sim_id'][p] != new_ind))
            better_than_new_s = np.logical_and.reduce((~new_better_than, ~H['local_pt'][p], H['sim_id'][p] != new_ind))

            # Who is closest to ind and better
            if np.any(better_than_new_l):
                ind = dist_to_all[better_than_new_l].argmin()
                H['ind_of_better_l'][new_ind] = H['sim_id'][p][np.nonzero(better_than_new_l)[0][ind]]
                H['dist_to_better_l'][new_ind] = dist_to_all[better_than_new_l][ind]

            if np.any(better_than_new_s):
                ind = dist_to_all[better_than_new_s].argmin()
                H['ind_of_better_s'][new_ind] = H['sim_id'][p][np.nonzero(better_than_new_s)[0][ind]]
                H['dist_to_better_s'][new_ind] = dist_to_all[better_than_new_s][ind]

            # if not ignore_L8:
            #     r_k = calc_rk(len(H['x_on_cube'][0]), n_s, rk_const, lhs_divisions)
            #     H['worse_within_rk'][new_ind][p] = np.logical_and.reduce((H['f'][new_ind] <= H['f'][p], dist_to_all <= r_k))

            #     # Add trues if new point is 'worse_within_rk'
            #     inds_to_change = np.logical_and.reduce((H['dist_to_all'][p,new_ind] <= r_k, H['f'][new_ind] >= H['f'][p], H['sim_id'][p] != new_ind))
            #     H['worse_within_rk'][inds_to_change,new_ind] = True

            #     if not H['local_pt'][new_ind]:
            #         H['worse_within_rk'][H['dist_to_all'] > r_k] = False


def update_history_optimal(x_opt, H, run_inds):
    """
    Updated the history after any point has been declared a local minimum
    """

    # opt_ind = np.where(np.logical_and(np.equal(x_opt,H['x_on_cube']).all(1),~np.isinf(H['f'])))[0] # This fails on some problems. x_opt is 1e-16 away from the point that was given and opt_ind is empty
    run_inds = np.unique(run_inds)

    dists = np.linalg.norm(H['x_on_cube'][run_inds]-x_opt, axis=1)
    ind = np.argmin(dists)
    opt_ind = run_inds[ind]

    tol_x1 = 1e-15
    if dists[ind] > tol_x1:
        print("Dist from reported x_opt to closest evaluated point is: " + str(dists[ind]) + "\n" +
              "Check that the local optimizer is working correctly\n", x_opt, run_inds, flush=True)

    assert dists[ind] <= tol_x1, "Closest point to x_opt not within {}?".format(tol_x1)

    tol_x2 = 1e-8
    failsafe = np.logical_and(H['f'][run_inds] < H['f'][opt_ind], dists < tol_x2)
    if np.any(failsafe):
        print("This run has {} point(s) with smaller 'f' value within {} of "
              "the point ruled to be the run minimum. \nMarking all as being "
              "a 'local_min' to prevent APOSMM from starting another run "
              "immediately from these points.".format(sum(failsafe), tol_x2))
        print("Sim_ids to be marked optimal: ", opt_ind, run_inds[failsafe])
        print("Check that the local optimizer is working correctly", flush=True)
        H['local_min'][run_inds[failsafe]] = 1

    H['local_min'][opt_ind] = 1
    H['num_active_runs'][run_inds] -= 1

    return opt_ind


def decide_where_to_start_localopt(H, n, n_s, rk_const, ld=0, mu=0, nu=0):
    """
    Finds points in the history that satisfy the conditions (S1-S5 and L1-L8) in
    Table 1 of the `APOSMM paper <https://doi.org/10.1007/s12532-017-0131-4>`_
    This method first identifies sample points satisfying S2-S5, and then
    identifies all localopt points that satisfy L1-L7.
    We then start from any sample point also satisfying S1.
    We do not check condition L8 currently.

    We don't consider points in the history that have not returned from
    computation, or that have a ``nan`` value. Also, note that ``mu`` and ``nu``
    implicitly depend on the scaling that is happening with the domain. That
    is, adjusting the initial domain can make a run start (or not start) at
    a point that didn't (or did) previously.

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    r_k_const: float
        Radius for deciding when to start runs
    lhs_divisions: integer
        Number of Latin hypercube sampling divisions (0 or 1 means uniform
        random sampling over the domain)
    mu: nonnegative float
        Distance from the boundary that all starting points must satisfy
    nu: nonnegative float
        Distance from identified minima that all starting points must satisfy
    gamma_quantile: float in (0,1]
        Only sample points whose function values are in the lower
        gamma_quantile can start localopt runs

    Returns
    ----------
    start_inds: list
        Indices where a local opt run should be started


    .. seealso::
        `start_persistent_local_opt_gens.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_persistent_local_opt_gens.py>`_
    """

    r_k = calc_rk(n, n_s, rk_const, ld)

    if nu > 0:
        test_2_through_5 = np.logical_and.reduce((
            H['returned'] == 1,  # have a returned function value
            H['dist_to_better_s'] >
            r_k,  # no better sample point within r_k (L2)
            ~H['started_run'],  # have not started a run (L3)
            H['dist_to_unit_bounds'] >=
            mu,  # have all components at least mu away from bounds (L4)
            np.all(
                cdist(H['x_on_cube'], H['x_on_cube'][H['local_min']]) >= nu,
                axis=1)  # distance nu away from known local mins (L5)
        ))
    else:
        test_2_through_5 = np.logical_and.reduce((
            H['returned'] == 1,  # have a returned function value
            H['dist_to_better_s'] >
            r_k,  # no better sample point within r_k (L2)
            ~H['started_run'],  # have not started a run (L3)
            H['dist_to_unit_bounds'] >=
            mu,  # have all components at least mu away from bounds (L4)
        ))  # (L5) is always true when nu = 0

    # assert gamma_quantile == 1, "This is not supported yet. What is the best way to decide this when there are NaNs present in H['f']?"
    # if gamma_quantile < 1:
    #     cut_off_value = np.sort(H['f'][~H['local_pt']])[np.floor(gamma_quantile*(sum(~H['local_pt'])-1)).astype(int)]
    # else:
    #     cut_off_value = np.inf

    # Find the indices of points that...
    sample_seeds = np.logical_and.reduce((
        ~H['local_pt'],  # are not localopt points
        # H['f'] <= cut_off_value,      # have a small enough objective value
        ~np.isinf(H['f']),  # have a non-infinity objective value
        ~np.isnan(H['f']),  # have a non-NaN objective value
        test_2_through_5,  # satisfy tests 2 through 5
    ))

    # Uncomment the following to test the effect of ignorning LocalOpt points
    # in APOSMM. This allows us to test a parallel MLSL.
    # return list(np.ix_(sample_seeds)[0])

    those_satisfying_S1 = H['dist_to_better_l'][sample_seeds] > r_k  # no better localopt point within r_k
    sample_start_inds = np.ix_(sample_seeds)[0][those_satisfying_S1]

    # Find the indices of points that...
    local_seeds = np.logical_and.reduce((
        H['local_pt'],  # are localopt points
        H['dist_to_better_l'] > r_k,  # no better local point within r_k (L1)
        ~np.isinf(H['f']),  # have a non-infinity objective value
        ~np.isnan(H['f']),  # have a non-NaN objective value
        test_2_through_5,
        H['num_active_runs'] == 0,  # are not in an active run (L6)
        ~H['local_min']  # are not a local min (L7)
    ))

    local_start_inds2 = list(np.ix_(local_seeds)[0])

    # If paused is a field in H, don't start from paused points.
    if 'paused' in H.dtype.names:
        sample_start_inds = sample_start_inds[~H[sample_start_inds]['paused']]
    start_inds = list(sample_start_inds)+local_start_inds2

    # Sort the starting inds by their function value
    inds = np.argsort(H['f'][start_inds])
    start_inds = np.array(start_inds)[inds].tolist()

    return start_inds


def calc_rk(n, n_s, rk_const, lhs_divisions=0):
    """ Calculate the critical distance r_k """

    if lhs_divisions == 0:
        r_k = rk_const*(log(n_s)/n_s)**(1/n)
    else:
        k = np.floor(n_s/lhs_divisions).astype(int)
        if k <= 1:  # to prevent r_k=0
            r_k = np.inf
        else:
            r_k = rk_const*(log(k)/k)**(1/n)

    return r_k


def initialize_APOSMM(H, user_specs, libE_info):
    """
    Computes common values every time that APOSMM is reinvoked

    .. seealso::
        `start_persistent_local_opt_gens.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_persistent_local_opt_gens.py>`_
    """
    n = len(user_specs['ub'])

    rk_c = user_specs.get('rk_const', ((gamma(1+(n/2.0))*5.0)**(1.0/n))/sqrt(pi))
    ld = user_specs.get('lhs_divisions', 0)
    mu = user_specs.get('mu', 1e-4)
    nu = user_specs.get('nu', 0)

    comm = libE_info['comm']

    local_H_fields = [('f', float),
                      ('grad', float, n),
                      ('x', float, n),
                      ('x_on_cube', float, n),
                      ('local_pt', bool),
                      ('known_to_aposmm', bool),
                      ('dist_to_unit_bounds', float),
                      ('dist_to_better_l', float),
                      ('dist_to_better_s', float),
                      ('ind_of_better_l', int),
                      ('ind_of_better_s', int),
                      ('started_run', bool),
                      ('num_active_runs', int),
                      ('local_min', bool),
                      ('sim_id', int),
                      ('paused', bool),
                      ('returned', bool),
                      ]

    if 'components' in user_specs:
        local_H_fields += [('fvec', float, user_specs['components'])]

    local_H = np.zeros(len(H), dtype=local_H_fields)

    if len(H):
        for field in H.dtype.names:
            local_H[field][:len(H)] = H[field]

        if user_specs['localopt_method'] in ['LD_MMA', 'blmvm']:
            assert 'grad' in H.dtype.names, "Must give 'grad' values to persistent_aposmm in gen_specs['in'] when using 'localopt_method'" + user_specs['localopt_method']
            assert not np.all(local_H['grad'] == 0), "All 'grad' values are zero for the given points."

        assert 'f' in H.dtype.names, "Must give 'f' values to persistent_aposmm in gen_specs['in']"
        assert 'sim_id' in H.dtype.names, "Must give 'sim_id' to persistent_aposmm in gen_specs['in']"
        assert 'returned' in H.dtype.names, "Must give 'returned' status to persistent_aposmm in gen_specs['in']"

        over_written_fields = ['dist_to_unit_bounds', 'dist_to_better_l', 'dist_to_better_s', 'ind_of_better_l', 'ind_of_better_s']
        if any([i in H.dtype.names for i in over_written_fields]):
            print("\n persistent_aposmm ignores any given values in these fields: " + str(over_written_fields) + "\n")

        initialize_dists_and_inds(local_H, len(H))

        # Update after receiving initial points
        update_history_dist(local_H, n)

    n_s = np.sum(~local_H['local_pt'])

    return n, n_s, rk_c, ld, mu, nu, comm, local_H


def initialize_children(user_specs):
    """ Initialize stuff for localopt children """
    local_opters = {}
    sim_id_to_child_inds = {}
    run_order = {}
    run_pts = {}  # This can differ from 'x_on_cube' if, for example, user_specs['periodic'] is True and run points are off the cube.
    total_runs = 0
    if user_specs['localopt_method'] in ['LD_MMA', 'blmvm']:
        fields_to_pass = ['x_on_cube', 'f', 'grad']
    elif user_specs['localopt_method'] in ['LN_SBPLX', 'LN_BOBYQA', 'LN_COBYLA', 'LN_NEWUOA',
                                           'LN_NELDERMEAD', 'scipy_Nelder-Mead', 'scipy_COBYLA',
                                           'external_localopt', 'nm']:
        fields_to_pass = ['x_on_cube', 'f']
    elif user_specs['localopt_method'] in ['pounders', 'dfols']:
        fields_to_pass = ['x_on_cube', 'fvec']
    else:
        raise NotImplementedError("Unknown local optimization method " "'{}'.".format(user_specs['localopt_method']))

    return local_opters, sim_id_to_child_inds, run_order, run_pts, total_runs, fields_to_pass


def add_k_sample_points_to_local_H(k, user_specs, persis_info, n, comm, local_H, sim_id_to_child_inds):

    if 'sample_points' in user_specs:
        v = np.sum(~local_H['local_pt'])  # Number of sample points so far
        sampled_points = user_specs['sample_points'][v:v+k]
        on_cube = False  # Assume points are on original domain, not unit cube
        if len(sampled_points):
            add_to_local_H(local_H, sampled_points, user_specs, on_cube=on_cube)
        k = k-len(sampled_points)

    if k > 0:
        sampled_points = persis_info['rand_stream'].uniform(0, 1, (k, n))
        add_to_local_H(local_H, sampled_points, user_specs, on_cube=True)

    return persis_info


def clean_up_and_stop(local_H, local_opters, run_order):
    # FIXME: This has to be a clean exit.

    print('[Parent]: The optimal points and values are:\n',
          local_H[np.where(local_H['local_min'])][['x', 'f']], flush=True)

    for i, p in local_opters.items():
        p.destroy(local_H['x_on_cube'][run_order[i][-1]])


# def display_exception(e):
#     print(e.__doc__)
#     print(e.args)
#     _, _, tb = sys.exc_info()
#     traceback.print_tb(tb)  # Fixed format
#     tb_info = traceback.extract_tb(tb)
#     filename, line, func, text = tb_info[-1]
#     print('An error occurred on line {} of function {} with statement {}'.format(line, func, text))

#     # PETSc/TAO errors are printed in the following manner:
#     if hasattr(e, '_traceback_'):
#         print('The error was:')
#         for i in e._traceback_:
#             print(i)
#     sys.stdout.flush()


# if __name__ == "__main__":
#     [H,gen_specs,persis_info] = [np.load('H20.npz')[i] for i in ['H','gen_specs','persis_info']]
#     gen_specs = gen_specs.item()
#     persis_info = persis_info.item()
#     import ipdb; ipdb.set_trace()
#     aposmm_logic(H,persis_info,gen_specs,{})

# vim:fdm=marker
