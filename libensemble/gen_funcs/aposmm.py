"""
This module contains methods used our implementation of the Asynchronously
Parallel Optimization Solver for finding Multiple Minima (APOSMM) method
described in detail in the paper
`https://doi.org/10.1007/s12532-017-0131-4 <https://doi.org/10.1007/s12532-017-0131-4>`_
"""
from __future__ import division
from __future__ import absolute_import

__all__ = ['aposmm_logic','initialize_APOSMM', 'decide_where_to_start_localopt', 'update_history_dist']

import sys, os, traceback
import numpy as np
# import scipy as sp
from scipy.spatial.distance import cdist, pdist, squareform
# from scipy import optimize as scipy_optimize

from mpi4py import MPI

from numpy.lib.recfunctions import merge_arrays

from math import log, gamma, pi, sqrt

from petsc4py import PETSc
import nlopt

def aposmm_logic(H,persis_info,gen_specs,_):
    """
    APOSMM as a libEnsemble generation function. Coordinates multiple local
    optimization runs, starting from points which do not have a better point
    nearby them. This generation function produces/requires the following
    fields in ``H``:

    - ``'x' [n floats]``: Parameters being optimized over
    - ``'x_on_cube' [n floats]``: Parameters scaled to the unit cube
    - ``'f' [float]``: Objective function being minimized
    - ``'local_pt' [bool]``: True if point from a local optimization run, false if it is a sample point
    - ``'dist_to_unit_bounds' [float]``: Distance to domain boundary
    - ``'dist_to_better_l' [float]``: Distance to closest better local optimization point
    - ``'dist_to_better_s' [float]``: Distance to closest better sample optimization point
    - ``'ind_of_better_l' [int]``: Index of point ``'dist_to_better_l``' away
    - ``'ind_of_better_s' [int]``: Index of point ``'dist_to_better_s``' away
    - ``'started_run' [bool]``: True if point has started a local optimization run
    - ``'num_active_runs' [int]``: Counts number of non-terminated local runs the point is in
    - ``'local_min' [float]``: True if point has been ruled a local minima

    and optionally

    - ``'priority' [float]``: Value quantifying a point's desirability
    - ``'f_i' [float]``: Value of ith objective component (if calculated one at a time)
    - ``'fvec' [m floats]``: All objective components (if calculated together)
    - ``'obj_component' [int]``: Index corresponding to value in ``'f_i``'
    - ``'pt_id' [int]``: Identify the point

    When using libEnsemble to do individual objective component evaluations,
    APOSMM will return ``gen_specs['components']`` copies of each point, but
    each component=0 version of the point will only be considered when

    - deciding where to start a run,
    - best nearby point,
    - storing the order of the points is the run
    - storing the combined objective function value
    - etc

    Necessary quantities in ``gen_specs`` are:

    - ``'lb' [n floats]``: Lower bound on search domain
    - ``'ub' [n floats]``: Upper bound on search domain
    - ``'initial_sample_size' [int]``: Number of uniformly sampled points that must be returned (with a non-nan value) before a local optimization run is started.
    - ``'localopt_method' [str]``: Name of an NLopt or PETSc/TAO method

    Optional ``gen_specs`` entries are:

    - ``'sample_points' [int]``: The points to be sampled (in the original domain)
    - ``'combine_component_func' [func]``: Function to combine objective components
    - ``'components' [int]``: Number of objective components
    - ``'dist_to_bound_multiple' [float in (0,1]]``: What fraction of the distance to the nearest boundary should the initial step size be in localopt runs
    - ``'high_priority_to_best_localopt_runs': [bool]``: True if localopt runs with smallest observed function value are given priority
    - ``'lhs_divisions' [int]``: Number of Latin hypercube sampling partitions (0 or 1 results in uniform sampling)
    - ``'min_batch_size' [int]``: Lower bound on the number of points given every time APOSMM is called
    - ``'mu' [float]``: Distance from the boundary that all localopt starting points must satisfy
    - ``'nu' [float]``: Distance from identified minima that all starting points must satisfy
    - ``'single_component_at_a_time' [bool]``: True if single objective components will be evaluated at a time
    - ``'rk_const' [float]``:

    And ``gen_specs`` convergence tolerances for NLopt and PETSc/TAO:

    - ``'fatol' [float]``:
    - ``'ftol_abs' [float]``:
    - ``'ftol_rel' [float]``:
    - ``'gatol' [float]``:
    - ``'grtol' [float]``:
    - ``'xtol_abs' [float]``:
    - ``'xtol_rel' [float]``:

    :Note:
        ``gen_specs['combine_component_func']`` must be defined when there are
        multiple objective components.

    :Note:
        APOSMM critically uses ``persis_info`` to store information about
        active runs, order of points in each run, etc. The allocation function
        must ensure it's always given.

    :See:
        ``libensemble/tests/regression_tests/test_branin_aposmm.py``
        for basic APOSMM usage.
    :See:
        ``libensemble/tests/regression_tests/test_chwirut_aposmm_one_residual_at_a_time.py``
        for an example of APOSMM coordinating multiple local optimization runs
        for an objective with more than one component.
    """

    """
    Description of intermediate variables in aposmm_logic:

    n:                domain dimension
    c_flag:           True if giving libEnsemble individual components of fvec to evaluate. (Note if c_flag is True, APOSMM will only use the com
    n_s:              the number of complete evaluations (not just component evaluations)
    updated_inds:     indices of H that have been updated (and so all their information must be sent back to libE manager to update)
    O:                new points to be sent back to the history


    x_new:            when re-running a local opt method to get the next point: stores the first new point requested by a local optimization method
    pt_in_run:        when re-running a local opt method to get the next point: counts function evaluations to know when a new point is given
    total_pts_in_run: when re-running a local opt method to get the next point: total evaluations in run to be incremented

    starting_inds:    indices where a runs should be started.
    active_runs:      indices of active local optimization runs (currently saved to disk between calls to APOSMM)
    sorted_run_inds:  indices of the considered run (in the order they were requested by the localopt method)
    x_opt:            the reported minimum from a localopt run (disregarded unless exit_code isn't 0)
    exit_code:        0 if a new localopt point has been found, otherwise it's the NLopt/POUNDERS code
    samples_needed:   counts the number of additional uniformly drawn samples needed

    """

    n, n_s, c_flag, O, r_k, mu, nu = initialize_APOSMM(H, gen_specs)

    # np.savez('H'+str(len(H)),H=H,gen_specs=gen_specs,persis_info=persis_info)
    if n_s < gen_specs['initial_sample_size']:
        updated_inds = set()

    else:
        global x_new, pt_in_run, total_pts_in_run # Used to generate a next local opt point

        updated_inds = update_history_dist(H, gen_specs, c_flag)

        starting_inds = decide_where_to_start_localopt(H, r_k, mu, nu)
        updated_inds.update(starting_inds)

        for ind in starting_inds:
            # Find the run number
            if not np.any(H['started_run']):
                persis_info['active_runs'] = set()
                persis_info['run_order'] = {}
                persis_info['old_runs'] = {}
                persis_info['total_runs'] = 0

            new_run_num = persis_info['total_runs']

            H['started_run'][ind] = 1
            H['num_active_runs'][ind] += 1

            persis_info['run_order'][new_run_num] = [ind]
            persis_info['active_runs'].update([new_run_num])
            persis_info['total_runs'] +=1

        num_runs = len(persis_info['run_order'])
        if 'max_active_runs' in gen_specs and gen_specs['max_active_runs'] < num_runs:
            run_vals = np.zeros((num_runs,2),dtype=int)
            for i,run in enumerate(persis_info['run_order'].keys()):
                run_vals[i,0] = run
                run_vals[i,1] = persis_info['run_order'][run][np.nanargmin(H['f'][persis_info['run_order'][run]])]

            P = squareform(pdist(H['x_on_cube'][run_vals[:,1]], 'euclidean'))
            dist_to_better = np.inf*np.ones(num_runs)

            for i in range(num_runs):
                better = H['f'][run_vals[:,1]]<H['f'][run_vals[i,1]]
                if any(better):
                    dist_to_better[i] = np.min(P[i,better])

            k_sorted = np.argpartition(-dist_to_better,kth=gen_specs['max_active_runs']-1) # Take max_active_runs largest
            
            persis_info['active_runs'] = set(run_vals[k_sorted[:gen_specs['max_active_runs']],0].astype(int))

        inactive_runs = set()

        # Find next point in any uncompleted runs using information stored in persis_info
        for run in persis_info['active_runs']:

            x_opt, exit_code, persis_info, sorted_run_inds = advance_localopt_method(H, gen_specs, c_flag, run, persis_info)

            if np.isinf(x_new).all():
                assert exit_code>0, "Exit code not zero, but no information in x_new.\n Local opt run " + str(run) + " after " + str(len(sorted_run_inds)) + " evaluations.\n Worker crashing!"
                # No new point was added. Hopefully at a minimum
                update_history_optimal(x_opt, H, sorted_run_inds)
                inactive_runs.add(run)
                updated_inds.update(sorted_run_inds)

            else:
                matching_ind = np.where(np.equal(x_new,O['x_on_cube']).all(1))[0]
                if len(matching_ind) == 0:
                    persis_info = add_points_to_O(O, x_new, H, gen_specs, c_flag, persis_info, local_flag=1, sorted_run_inds=sorted_run_inds, run=run)
                else:
                    assert len(matching_ind) == 1, "This point shouldn't have ended up in the O twice!"
                    persis_info['run_order'][run].append(O['sim_id'][matching_ind[0]])

        for i in inactive_runs:
            persis_info['active_runs'].remove(i)
            old_run = persis_info['run_order'].pop(i) # Deletes any information about this run
            persis_info['old_runs'][i] = old_run

    if len(H) == 0:
        samples_needed = gen_specs['initial_sample_size']
    elif 'min_batch_size' in gen_specs:
        samples_needed = gen_specs['min_batch_size'] - len(O)
    else:
        samples_needed = int(not bool(len(O))) # 1 if len(O)==0, 0 otherwise

    if samples_needed > 0:
        if 'sample_points' in gen_specs:
            v = sum(H['local_pt'])
            x_new = gen_specs['sample_points'][v:v+samples_needed]
            on_cube = False # We assume the points are on the original domain, not unit cube
        else:
            x_new = persis_info['rand_stream'].uniform(0,1,(samples_needed,n))
            on_cube = True

        persis_info = add_points_to_O(O, x_new, H, gen_specs, c_flag, persis_info, on_cube=on_cube)

    O = np.append(H[np.array(list(updated_inds),dtype=int)][[o[0] for o in gen_specs['out']]],O)

    return O, persis_info

def add_points_to_O(O, pts, H, gen_specs, c_flag, persis_info, local_flag=0, sorted_run_inds=[], run=[], on_cube=True):
    """
    Adds points to O, the numpy structured array to be sent back to the manager
    """

    assert not local_flag or len(pts) == 1, "add_points_to_O does not support this functionality"

    original_len_O = len(O)

    len_H = len(H)
    ub = gen_specs['ub']
    lb = gen_specs['lb']
    if c_flag:
        m = gen_specs['components']

        assert len_H % m == 0, "Number of points in len_H not congruent to 0 mod 'components'"
        pt_ids = np.sort(np.tile(np.arange((len_H+original_len_O)/m,(len_H+original_len_O)/m + len(pts)),(1,m)))
        pts = np.tile(pts,(m,1))

    num_pts = len(pts)

    O.resize(len(O)+num_pts,refcheck=False) # Adds (num_pts) rows of zeros to O

    if on_cube:
        O['x_on_cube'][-num_pts:] = pts
        O['x'][-num_pts:] = pts*(ub-lb)+lb
    else:
        O['x_on_cube'][-num_pts:] = (pts-lb)/(ub-lb)
        O['x'][-num_pts:] = pts

    O['sim_id'][-num_pts:] = np.arange(len_H+original_len_O,len_H+original_len_O+num_pts)
    O['local_pt'][-num_pts:] = local_flag

    O['dist_to_unit_bounds'][-num_pts:] = np.inf
    O['dist_to_better_l'][-num_pts:] = np.inf
    O['dist_to_better_s'][-num_pts:] = np.inf
    O['ind_of_better_l'][-num_pts:] = -1
    O['ind_of_better_s'][-num_pts:] = -1

    if c_flag:
        O['obj_component'][-num_pts:] = np.tile(range(0,m),(1,num_pts//m))
        O['pt_id'][-num_pts:] = pt_ids

    if local_flag:
        O['num_active_runs'][-num_pts] += 1
        # O['priority'][-num_pts:] = 1
        # O['priority'][-num_pts:] = np.random.uniform(0,1,num_pts)
        if 'high_priority_to_best_localopt_runs' in gen_specs and gen_specs['high_priority_to_best_localopt_runs']:
            O['priority'][-num_pts:] = -min(H['f'][persis_info['run_order'][run]]) # Give highest priority to run with lowest function value
        else:
            O['priority'][-num_pts:] = persis_info['rand_stream'].uniform(0,1,num_pts)
        persis_info['run_order'][run].append(O[-num_pts]['sim_id'])
    else:
        if c_flag:
            # p_tmp = np.sort(np.tile(np.random.uniform(0,1,num_pts/m),(m,1))) # If you want all "duplicate points" to have the same priority (meaning libEnsemble gives them all at once)
            # p_tmp = np.random.uniform(0,1,num_pts)
            p_tmp = persis_info['rand_stream'].uniform(0,1,num_pts)
        else:
            # p_tmp = np.random.uniform(0,1,num_pts)
            # persis_info['rand_stream'].uniform(lb,ub,(1,n))
            if 'high_priority_to_best_localopt_runs' in gen_specs and gen_specs['high_priority_to_best_localopt_runs']:
                p_tmp = -np.inf*np.ones(num_pts)
            else:
                p_tmp = persis_info['rand_stream'].uniform(0,1,num_pts)
        O['priority'][-num_pts:] = p_tmp
        # O['priority'][-num_pts:] = 1

    return persis_info


def update_history_dist(H, gen_specs, c_flag):
    """
    Updates distances/indices after new points that have been evaluated.

    :See:
        ``/libensemble/alloc_funcs/start_persistent_local_opt_gens.py``
    """

    n = len(H['x_on_cube'][0])

    updated_inds = set()

    new_inds = np.where(~H['known_to_aposmm'])[0]

    if c_flag:
        for v in np.unique(H['pt_id'][new_inds]):
            inds = H['pt_id']==v
            H['f'][inds] = np.inf
            H['f'][np.where(inds)[0][0]] = gen_specs['combine_component_func'](H['f_i'][inds])

        p = np.logical_and.reduce((H['returned'],H['obj_component']==0,~np.isnan(H['f'])))
    else:
        p = np.logical_and.reduce((H['returned'],~np.isnan(H['f'])))


    for new_ind in new_inds:
        # Loop over new returned points and update their distances
        if p[new_ind]:
            H['known_to_aposmm'][new_ind] = True # These points are now known to APOSMM

            # Compute distance to boundary
            H['dist_to_unit_bounds'][new_ind] = min(min(np.ones(n) - H['x_on_cube'][new_ind]),min(H['x_on_cube'][new_ind] - np.zeros(n)))

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
            updated_inds.update(updates)

            # Since we allow equality when deciding better_than_new_l and
            # better_than_new_s, we have to prevent new_ind from being its own
            # better point.
            better_than_new_l = np.logical_and.reduce((~new_better_than,  H['local_pt'][p], H['sim_id'][p] != new_ind))
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

    updated_inds.update(new_inds)
    return updated_inds



def update_history_optimal(x_opt, H, run_inds):
    """
    Updated the history after any point has been declared a local minimum
    """

    opt_ind = np.where(np.logical_and(np.equal(x_opt,H['x_on_cube']).all(1),~np.isinf(H['f'])))[0]
    assert len(opt_ind) == 1, "Why isn't there exactly one optimal point?"
    assert opt_ind in run_inds, "Why isn't the run optimum a point in the run?"

    H['local_min'][opt_ind] = 1
    H['num_active_runs'][run_inds] -= 1



def advance_localopt_method(H, gen_specs, c_flag, run, persis_info):
    """
    Moves a local optimization method one iteration forward. We currently do
    this by feeding all past evaluations from a run to the method and then
    storing the first new point generated
    """

    global x_new, pt_in_run, total_pts_in_run # Used to generate a next local opt point

    while 1:
        sorted_run_inds = persis_info['run_order'][run]
        assert all(H['returned'][sorted_run_inds])

        x_new = np.ones((1,len(gen_specs['ub'])))*np.inf; pt_in_run = 0; total_pts_in_run = len(sorted_run_inds)

        if gen_specs['localopt_method'] in ['LN_SBPLX', 'LN_BOBYQA', 'LN_NELDERMEAD', 'LD_MMA']:

            if gen_specs['localopt_method'] in ['LD_MMA']:
                fields_to_pass = ['x_on_cube','f','grad']
            else:
                fields_to_pass = ['x_on_cube','f']

            try:
                x_opt, exit_code = set_up_and_run_nlopt(H[fields_to_pass][sorted_run_inds], gen_specs)
            except Exception as e:
                exit_code = 0
                display_exception(e)

        elif gen_specs['localopt_method'] in ['pounders']:

            if c_flag:
                Run_H_F = np.zeros(len(sorted_run_inds),dtype=[('fvec',float,gen_specs['components'])])
                for i,ind in enumerate(sorted_run_inds):
                    for j in range(gen_specs['components']):
                        Run_H_F['fvec'][i][j] = H['f_i'][np.logical_and(H['pt_id']==H['pt_id'][ind], H['obj_component']==j)]
                Run_H = merge_arrays([H[['x_on_cube']][sorted_run_inds],Run_H_F],flatten=True)
            else:
                Run_H = H[['x_on_cube','fvec']][sorted_run_inds]

            try:
                x_opt, exit_code = set_up_and_run_tao(Run_H, gen_specs)
            except Exception as e:
                exit_code = 0
                display_exception(e)

        # elif gen_specs['localopt_method'] in ['COBYLA']:

        #     fields_to_pass = ['x_on_cube','f']

        #     try:
        #         x_opt, exit_code = set_up_and_run_scipy_minimize(H[fields_to_pass][sorted_run_inds], gen_specs)
        #     except Exception as e:
        #         exit_code = 0
        #         display_exception(e)


        else:
            sys.exit("Unknown localopt method. Exiting")

        matching_ind = np.equal(x_new,H['x_on_cube']).all(1)
        if ~matching_ind.any():
            # Generated a new point
            break
        else:
            # We need to add a previously evaluated point into this run
            persis_info['run_order'][run].append(np.nonzero(matching_ind)[0][0])

    return x_opt, exit_code, persis_info, sorted_run_inds



# def set_up_and_run_scipy_minimize(Run_H, gen_specs):
#     """ Set up objective and runs scipy

#     Declares the appropriate syntax for our special objective function to read
#     through Run_H, sets the parameters and starting points for the run.
#     """

#     def scipy_obj_fun(x, Run_H):
#         out = look_in_history(x, Run_H)

#         return out

#     obj = lambda x: scipy_obj_fun(x, Run_H)
#     x0 = Run_H['x_on_cube'][0]


#     import ipdb; ipdb.set_trace()
#     #construct the bounds in the form of constraints
#     cons = []
#     for factor in range(len(x0)):
#         l = {'type': 'ineq',
#              'fun': lambda x, lb=gen_specs['lb'][factor], i=factor: x[i] - lb}
#         u = {'type': 'ineq',
#              'fun': lambda x, ub=gen_specs['ub'][factor], i=factor: ub - x[i]}
#         cons.append(l)
#         cons.append(u)

#     res = scipy_optimize.minimize(obj,x0,method=gen_specs['localopt_method'],options={'maxiter':len(Run_H['x_on_cube'])+1})

#     if res['status'] == 2: # SciPy code for exhausting budget of evaluations, so not at a minimum
#         exit_code = 0

#     x_opt = res['x']
#     return x_opt, exit_code


def set_up_and_run_nlopt(Run_H, gen_specs):
    """ Set up objective and runs nlopt

    Declares the appropriate syntax for our special objective function to read
    through Run_H, sets the parameters and starting points for the run.
    """

    assert 'xtol_rel' or 'xtol_abs' or 'ftol_rel' or 'ftol_abs' in gen_specs, "NLopt can cycle if xtol_rel, xtol_abs, ftol_rel, or ftol_abs are not set"

    def nlopt_obj_fun(x, grad, Run_H):
        out = look_in_history(x, Run_H)

        if gen_specs['localopt_method'] in ['LD_MMA']:
            grad[:] = out[1]
            out = out[0]

        return out

    n = len(gen_specs['ub'])

    opt = nlopt.opt(getattr(nlopt,gen_specs['localopt_method']), n)

    lb = np.zeros(n)
    ub = np.ones(n)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    x0 = Run_H['x_on_cube'][0]

    # Care must be taken here because a too-large initial step causes nlopt to move the starting point!
    dist_to_bound = min(min(ub-x0),min(x0-lb))
    assert dist_to_bound > np.finfo(np.float32).eps, "The distance to the boundary is too small for NLopt to handle"

    if 'dist_to_bound_multiple' in gen_specs:
        opt.set_initial_step(dist_to_bound*gen_specs['dist_to_bound_multiple'])
    else:
        opt.set_initial_step(dist_to_bound)

    opt.set_maxeval(len(Run_H)+1) # evaluate one more point
    opt.set_min_objective(lambda x, grad: nlopt_obj_fun(x, grad, Run_H))
    if 'xtol_rel' in gen_specs:
        opt.set_xtol_rel(gen_specs['xtol_rel'])
    if 'ftol_rel' in gen_specs:
        opt.set_ftol_rel(gen_specs['ftol_rel'])
    if 'xtol_abs' in gen_specs:
        opt.set_xtol_abs(gen_specs['xtol_abs'])
    if 'ftol_abs' in gen_specs:
        opt.set_ftol_abs(gen_specs['ftol_abs'])

    x_opt = opt.optimize(x0)
    exit_code = opt.last_optimize_result()

    if exit_code == 5: # NLOPT code for exhausting budget of evaluations, so not at a minimum
        exit_code = 0

    return x_opt, exit_code


def set_up_and_run_tao(Run_H, gen_specs):
    """ Set up objective and runs PETSc on the comm_self communicator

    Declares the appropriate syntax for our special objective function to read
    through Run_H, sets the parameters and starting points for the run.
    """
    tao_comm = MPI.COMM_SELF
    n = len(gen_specs['ub'])
    m = len(Run_H['fvec'][0])

    def pounders_obj_func(tao, X, F, Run_H):
        F.array = look_in_history(X.array, Run_H, vector_return=True)
        return F

    # def blmvm_obj_func(tao, X, G, Run_H):
    #     (f, grad) = look_in_history_fd_grad(X.array, Run_H)
    #     G.array = grad
    #     return f

    # Create starting point, bounds, and tao object
    x = PETSc.Vec().create(tao_comm)
    x.setSizes(n)
    x.setFromOptions()
    x.array = Run_H['x_on_cube'][0]
    lb = x.duplicate()
    ub = x.duplicate()
    lb.array = 0*np.ones(n)
    ub.array = 1*np.ones(n)
    tao = PETSc.TAO().create(tao_comm)
    tao.setType(gen_specs['localopt_method'])

    # if gen_specs['localopt_method'] == 'pounders':
    f = PETSc.Vec().create(tao_comm)
    f.setSizes(m)
    f.setFromOptions()

    delta_0 = gen_specs['dist_to_bound_multiple']*np.min([np.min(ub.array-x.array), np.min(x.array-lb.array)])

    PETSc.Options().setValue('-tao_pounders_delta',str(delta_0))
    # PETSc.Options().setValue('-pounders_subsolver_tao_type','bqpip')
    tao.setSeparableObjective(lambda tao, x, f: pounders_obj_func(tao, x, f, Run_H), f)
    # elif gen_specs['localopt_method'] == 'blmvm':
    #     g = PETSc.Vec().create(tao_comm)
    #     g.setSizes(n)
    #     g.setFromOptions()
    #     tao.setObjectiveGradient(lambda tao, x, g: blmvm_obj_func(tao, x, g, Run_H))

    # Set everything for tao before solving
    PETSc.Options().setValue('-tao_max_funcs',str(total_pts_in_run+1))
    tao.setFromOptions()
    tao.setVariableBounds((lb,ub))
    # tao.setObjectiveTolerances(fatol=gen_specs['fatol'], frtol=gen_specs['frtol'])
    # tao.setGradientTolerances(grtol=gen_specs['grtol'], gatol=gen_specs['gatol'])
    tao.setTolerances(grtol=gen_specs['grtol'], gatol=gen_specs['gatol'])
    tao.setInitial(x)

    tao.solve(x)

    x_opt = tao.getSolution().getArray()
    exit_code = tao.getConvergedReason()
    # print(exit_code)
    # print(tao.view())
    # print(x_opt)

    # if gen_specs['localopt_method'] == 'pounders':
    f.destroy()
    # elif gen_specs['localopt_method'] == 'blmvm':
    #     g.destroy()

    lb.destroy()
    ub.destroy()
    x.destroy()
    tao.destroy()

    return x_opt, exit_code



def decide_where_to_start_localopt(H, r_k, mu=0, nu=0, gamma_quantile=1):
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


    :See:
        ``/libensemble/alloc_funcs/start_persistent_local_opt_gens.py``
    """

    n = len(H['x_on_cube'][0])

    if nu > 0:
        test_2_through_5 = np.logical_and.reduce((
                H['returned'] == 1,          # have a returned function value
                H['dist_to_better_s'] > r_k, # no better sample point within r_k (L2)
               ~H['started_run'],            # have not started a run (L3)
                H['dist_to_unit_bounds'] >= mu, # have all components at least mu away from bounds (L4)
                np.all(cdist(H['x_on_cube'], H['x_on_cube'][H['local_min']]) >= nu,axis=1) # distance nu away from known local mins (L5)
            ))
    else:
        test_2_through_5 = np.logical_and.reduce((
                H['returned'] == 1,          # have a returned function value
                H['dist_to_better_s'] > r_k, # no better sample point within r_k (L2)
               ~H['started_run'],            # have not started a run (L3)
                H['dist_to_unit_bounds'] >= mu, # have all components at least mu away from bounds (L4)
            )) # (L5) is always true when nu = 0

    assert gamma_quantile == 1, "This is not supported yet. What is the best way to decide this when there are NaNs present in H['f']?"
    # if gamma_quantile < 1:
    #     cut_off_value = np.sort(H['f'][~H['local_pt']])[np.floor(gamma_quantile*(sum(~H['local_pt'])-1)).astype(int)]
    # else:
    #     cut_off_value = np.inf

    ### Find the indices of points that...
    sample_seeds = np.logical_and.reduce((
           ~H['local_pt'],               # are not localopt points
           # H['f'] <= cut_off_value,      # have a small enough objective value
           ~np.isinf(H['f']),            # have a non-infinity objective value
           ~np.isnan(H['f']),            # have a non-NaN objective value
           test_2_through_5,             # satisfy tests 2 through 5
         ))

    # Uncomment the following to test the effect of ignorning LocalOpt points
    # in APOSMM. This allows us to test a parallel MLSL.
    # return list(np.ix_(sample_seeds)[0])

    those_satisfying_S1 = H['dist_to_better_l'][sample_seeds] > r_k # no better localopt point within r_k
    sample_start_inds = np.ix_(sample_seeds)[0][those_satisfying_S1]

    ### Find the indices of points that...
    local_seeds = np.logical_and.reduce((
            H['local_pt'],               # are localopt points
            H['dist_to_better_l'] > r_k, # no better local point within r_k (L1)
           ~np.isinf(H['f']),            # have a non-infinity objective value
           ~np.isnan(H['f']),            # have a non-NaN objective value
            test_2_through_5,
            H['num_active_runs'] == 0,   # are not in an active run (L6)
           ~H['local_min'] # are not a local min (L7)
         ))


    local_start_inds2 = list(np.ix_(local_seeds)[0])

    start_inds = list(sample_start_inds) + local_start_inds2
    return start_inds

def look_in_history(x, Run_H, vector_return=False):
    """ See if Run['x_on_cube'][pt_in_run] matches x, returning f or fvec, or saves x to
    x_new if every point in Run_H has been checked.
    """

    global pt_in_run, total_pts_in_run, x_new

    if vector_return:
        to_return = 'fvec'
    else:
        if 'grad' in Run_H.dtype.names:
            to_return = ['f','grad']
        else:
            to_return = 'f'

    if pt_in_run < total_pts_in_run:
        # Return the value in history to the localopt algorithm.
        assert np.allclose(x, Run_H['x_on_cube'][pt_in_run], rtol=1e-08, atol=1e-08), \
            "History point does not match Localopt point"
        f_out = Run_H[to_return][pt_in_run]
    else:
        if pt_in_run == total_pts_in_run:
            # The history of points is exhausted. Save the requested point x to
            # x_new. x_new will be returned to the manager.
            x_new[:] = x

        # Just in case the local opt method requests more points after a new
        # point has been identified.
        # f_out = np.finfo(np.float64).max
        # f_out = Run_H[to_return][total_pts_in_run-1]
        f_out = Run_H[to_return][total_pts_in_run-1]

    pt_in_run += 1

    return f_out



def calc_rk(n, n_s, rk_const, lhs_divisions=0):
    """ Calculate the critical distance r_k """

    if lhs_divisions == 0:
        r_k = rk_const*(log(n_s)/n_s)**(1/n)
    else:
        k = np.floor(n_s/lhs_divisions).astype(int)
        if k <= 1: # to prevent r_k=0
            r_k = np.inf
        else:
            r_k = rk_const*(log(k)/k)**(1/n)

    return r_k

def initialize_APOSMM(H, gen_specs):
    """
    Computes common values every time that APOSMM is reinvoked

    :See:
        ``/libensemble/alloc_funcs/start_persistent_local_opt_gens.py``
    """

    n = len(gen_specs['ub'])

    if 'single_component_at_a_time' in gen_specs and gen_specs['single_component_at_a_time']:
        assert gen_specs['batch_mode'], "Must be in batch mode when using 'single_component_at_a_time' and APOSMM"
        c_flag = True
    else:
        c_flag = False


    if c_flag:
        pt_ids = H['pt_id'][np.logical_and(H['returned'],~np.isnan(H['f_i']))] # Get the pt_id for non-nan, returned points
        _, counts = np.unique(pt_ids,return_counts=True)
        n_s = np.sum(counts == gen_specs['components'])
    else:
        n_s = np.sum(np.logical_and.reduce((~np.isnan(H['f']),~H['local_pt'], H['returned']))) # Number of returned sampled points (excluding nans)

    # Rather than build up a large output, we will just make changes in the
    # given H, and then send back the rows corresponding to updated H entries.
    O = np.empty(0,dtype=gen_specs['out'])

    if 'rk_const' in gen_specs:
        rk_c = gen_specs['rk_const']
    else:
        rk_c = ((gamma(1+(n/2.0))*5.0)**(1.0/n))/sqrt(pi)

    if 'lhs_divisions' in gen_specs:
        ld = gen_specs['lhs_divisions']
    else:
        ld = 0

    if 'mu' in gen_specs:
        mu = gen_specs['mu']
    else:
        mu = 1e-4

    if 'nu' in gen_specs:
        nu = gen_specs['nu']
    else:
        nu = 0

    if n_s > 0:
        r_k = calc_rk(n, n_s, rk_c, ld)
    else:
        r_k = np.inf

    return n, n_s, c_flag, O, r_k, mu, nu


def queue_update_function(H, gen_specs, persis_info):
    """
    A specific queue update function that stops evaluations under a variety of
    conditions

    gen_specs['stop_on_NaNs']
    gen_specs['stop_partial_fvec_eval']
    H['paused']
    """

    if len(H)==persis_info['H_len']:
        return persis_info
    else:
        persis_info['H_len']=len(H)

    pt_ids_to_pause = set()

    # Pause entries in H if one component is evaluated at a time and there are
    # any NaNs for some components.
    if 'stop_on_NaNs' in gen_specs and gen_specs['stop_on_NaNs']:
        pt_ids_to_pause.update(H['pt_id'][np.isnan(H['f_i'])])

    # Pause entries in H if a partial combine_component_func evaluation is
    # worse than the best, known, complete evaluation (and the point is not a
    # local_opt point).
    if 'stop_partial_fvec_eval' in gen_specs and gen_specs['stop_partial_fvec_eval']:
        pt_ids = np.unique(H['pt_id'])

        complete_fvals_flag = np.zeros(len(pt_ids),dtype=bool)
        for i,pt_id in enumerate(pt_ids):
            if pt_id in persis_info['has_nan']:
                continue

            a1 = H['pt_id']==pt_id
            if np.any(np.isnan(H['f_i'][a1])):
                persis_info['has_nan'].add(pt_id)
                continue

            if np.all(H['returned'][a1]):
                complete_fvals_flag[i] = True
                persis_info['complete'].add(pt_id)

        # complete_fvals_flag = np.array([np.all(H['returned'][H['pt_id']==i]) for i in pt_ids],dtype=bool)

        if np.any(complete_fvals_flag) and len(pt_ids)>1:
            # Ensure combine_component_func calculates partial fevals correctly
            # with H['f_i'] = 0 for non-returned point
            possibly_partial_fvals = np.array([gen_specs['combine_component_func'](H['f_i'][H['pt_id']==i]) for i in pt_ids])

            best_complete = np.nanmin(possibly_partial_fvals[complete_fvals_flag])

            worse_flag = np.zeros(len(pt_ids),dtype=bool)
            for i in range(len(pt_ids)):
                if not np.isnan(possibly_partial_fvals[i]) and possibly_partial_fvals[i] > best_complete:
                    worse_flag[i] = True

            # Pause incompete evaluations with worse_flag==True
            pt_ids_to_pause.update(pt_ids[np.logical_and(worse_flag,~complete_fvals_flag)])

    if not pt_ids_to_pause.issubset(persis_info['already_paused']):
        persis_info['already_paused'].update(pt_ids_to_pause)
        H['paused'][np.in1d(H['pt_id'],list(pt_ids_to_pause))] = True

    return persis_info

def display_exception(e):
    print(e.__doc__)
    print(e.args)
    _, _, tb = sys.exc_info()
    traceback.print_tb(tb) # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]
    print('An error occurred on line {} in statement {}'.format(line, text))

# if __name__ == "__main__":
#     [H,gen_specs,persis_info] = [np.load('H20.npz')[i] for i in ['H','gen_specs','persis_info']]
#     gen_specs = gen_specs.item()
#     persis_info = persis_info.item()
#     import ipdb; ipdb.set_trace()
#     aposmm_logic(H,persis_info,gen_specs,{})
