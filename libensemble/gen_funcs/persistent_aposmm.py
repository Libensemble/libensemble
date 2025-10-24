"""
This module contains methods used our implementation of the Asynchronously
Parallel Optimization Solver for finding Multiple Minima (APOSMM) method.
`https://doi.org/10.1007/s12532-017-0131-4 <https://doi.org/10.1007/s12532-017-0131-4>`_

This implementation of APOSMM was developed by Kaushik Kulkarni and Jeffrey
Larson in the summer of 2019.
"""

__all__ = ["aposmm", "initialize_APOSMM", "decide_where_to_start_localopt", "update_history_dist"]

from math import log, pi, sqrt

import numpy as np
from mpmath import gamma

from libensemble.gen_funcs.aposmm_localopt_support import ConvergedMsg, LocalOptInterfacer, simulate_recv_from_manager
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

# from scipy.spatial.distance import cdist


# Due to recursion error in scipy cdist function
def cdist(XA, XB, metric="euclidean"):
    """Compute the pairwise Euclidean distances"""

    # Just so we don't have to change the call
    if metric != "euclidean":
        raise ValueError("Only 'euclidean' metric is supported in this implementation.")

    # Compute the pairwise Euclidean distances
    diff = XA[:, np.newaxis, :] - XB[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances


def aposmm(H, persis_info, gen_specs, libE_info):
    """
    APOSMM coordinates multiple local optimization runs, dramatically reducing time for
    discovering multiple minima on parallel systems. APOSMM tracks these fields:

    - ``"x" [n floats]``: Parameters being optimized over
    - ``"x_on_cube" [n floats]``: Parameters scaled to the unit cube
    - ``"f" [float]``: Objective function being minimized
    - ``"local_pt" [bool]``: True if point from a local optimization run
    - ``"started_run" [bool]``: True if point has started a local opt run
    - ``"num_active_runs" [int]``: Number of active local runs point is in
    - ``"local_min" [float]``: True if point has been ruled a local minima
    - ``"sim_id" [int]``: Row number of entry in history

    and optionally

    - ``"fvec" [m floats]``: All objective components (if performing a least-squares calculation)
    - ``"grad" [n floats]``: The gradient (if available) of the objective with respect to `x`.

    Note:

    - If any of the above fields are desired after a libEnsemble run, name
      them in ``gen_specs["out"]``.
    - If intitializing APOSMM with past function values, make sure to include
      ``"x"``, ``"x_on_cube"``, ``"f"``, ``"local_pt"``, etc. in
      ``gen_specs["in"]`` (and, of course, include them in the H0 array given
      to libensemble).

    Necessary quantities in ``gen_specs["user"]`` are:

    - ``"lb" [n floats]``: Lower bound on search domain
    - ``"ub" [n floats]``: Upper bound on search domain
    - ``"localopt_method" [str]``: Name of an NLopt, PETSc/TAO, or SciPy method
      (see "advance_local_run" below for supported methods). When using a SciPy
      method, must supply ``"opt_return_codes"``, a list of integers that will
      be used to determine if the x produced by the localopt method should be
      ruled a local minimum. (For example, SciPy's COBYLA has a "status" of ``1`` if
      at an optimum, but SciPy's Nelder-Mead and BFGS have a "status" of ``0`` if at
      an optimum.)
    - ``"initial_sample_size" [int]``: Number of uniformly sampled points
      to be evaluated before starting the localopt runs. Can be
      zero if no additional sampling is desired, but if zero there must be past
      sim_f values given to libEnsemble in H0.

    Optional ``gen_specs["user"]`` entries are:

    - ``"max_active_runs" [int]``: Bound on number of runs APOSMM is advancing
    - ``"sample_points" [numpy array]``: Points to be sampled (original domain).
      If more sample points are needed by APOSMM during the course of the
      optimization, points will be drawn uniformly over the domain
    - ``"components" [int]``: Number of objective components
    - ``"dist_to_bound_multiple" [float in (0, 1]]``: What fraction of the
      distance to the nearest boundary should the initial step size be in
      localopt runs
    - ``"lhs_divisions" [int]``: Number of Latin hypercube sampling partitions
      (0 or 1 results in uniform sampling)
    - ``"mu" [float]``: Distance from the boundary that all localopt starting
      points must satisfy
    - ``"nu" [float]``: Distance from identified minima that all starting
      points must satisfy
    - ``"rk_const" [float]``: Multiplier in front of the ``r_k`` value
    - ``"stop_after_k_minima" [int]``: Tell APOSMM to stop after this many
      local minima have been identified by a local optimization run.
    - ``"stop_after_k_runs" [int]``: Tell APOSMM to stop after this many runs
      have ended. (The number of ended runs may be less than the number of
      minima if, for example, a local optimization run ends due to a evaluation
      constraint, but not convergence criteria.)

    If the rules in ``decide_where_to_start_localopt`` produces more than
    ``"max_active_runs"`` in some iteration, then existing runs are prioritized.

    And ``gen_specs["user"]`` must also contain fields for the given
    localopt_method's convergence tolerances (e.g., ``gatol/grtol`` for PETSC/TAO
    or ``ftol_rel`` for NLopt)

    .. seealso::

        `test_persistent_aposmm_scipy <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_scipy.py>`_
        for basic APOSMM usage.

    .. seealso::

        `test_persistent_aposmm_with_grad <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_with_grad.py>`_
        for an example where past function values are given to libEnsemble/APOSMM.

    .. seealso::

        `test_aposmm_starting_point_finder <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/unit_tests/test_aposmm_starting_point_finder.py>`_
        for an example the APOSMM r_k radius logic is adjusted to produce a certain number of localopt starting points.

    """
    """
    Description of intermediate variables in aposmm:

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
                      unless opt_flag is 1)
    opt_flag:         1 if the run ended with an optimal point (x_opt) or
                      0 if it ended because e.g., maxiters/maxevals were reached
    num_samples:      Number of additional uniformly drawn samples needed


    Description of persistent variables used to maintain the state of APOSMM

    persis_info['total_runs']: Running count of started/completed localopt runs
    persis_info['run_order']: Sequence of indices of points in unfinished runs

    """

    try:
        user_specs = gen_specs["user"]
        ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
        n, n_s, rk_const, ld, mu, nu, comm, local_H = initialize_APOSMM(H, user_specs, libE_info)
        (
            local_opters,
            sim_id_to_child_inds,
            run_order,
            run_pts,
            total_runs,
            ended_runs,
            fields_to_pass,
        ) = initialize_children(user_specs)

        if user_specs["initial_sample_size"] != 0:
            # Send our initial sample. We don't need to check that n_s is large enough:
            # the alloc_func only returns when the initial sample has function values.
            persis_info = add_k_sample_points_to_local_H(
                user_specs["initial_sample_size"], user_specs, persis_info, n, comm, local_H, sim_id_to_child_inds
            )
            if not user_specs.get("standalone"):
                ps.send(local_H[-user_specs["initial_sample_size"] :][[i[0] for i in gen_specs["out"]]])
            something_sent = True
        else:
            something_sent = False

        tag = None
        first_pass = True
        while 1:
            new_opt_inds_to_send_mgr = []
            new_inds_to_send_mgr = []

            if something_sent:
                if user_specs.get("standalone"):
                    tag, Work, calc_in = simulate_recv_from_manager(local_H, gen_specs)
                else:
                    tag, Work, calc_in = ps.recv()

                if tag in [STOP_TAG, PERSIS_STOP]:
                    clean_up_and_stop(local_opters)
                    persis_info["run_order"] = run_order
                    break

                if np.sum(local_H["local_min"]) >= user_specs.get("stop_after_k_minima", np.inf) or len(
                    ended_runs
                ) >= user_specs.get("stop_after_k_runs", np.inf):
                    # This break happens here so the manager can be informed about the last minima.
                    clean_up_and_stop(local_opters)
                    persis_info["run_order"] = run_order
                    break

                n_s, n_r = update_local_H_after_receiving(local_H, n, n_s, user_specs, Work, calc_in, fields_to_pass)

                for row in calc_in:
                    if sim_id_to_child_inds.get(row["sim_id"]):
                        # Point came from a child local opt run
                        for child_idx in sim_id_to_child_inds[row["sim_id"]]:
                            x_new = local_opters[child_idx].iterate(row[fields_to_pass])
                            if isinstance(x_new, ConvergedMsg):
                                x_opt = x_new.x
                                opt_flag = x_new.opt_flag
                                opt_ind = update_history_optimal(x_opt, opt_flag, local_H, run_order[child_idx])
                                new_opt_inds_to_send_mgr.append(opt_ind)
                                local_opters.pop(child_idx)
                                ended_runs.append(child_idx)
                            else:
                                add_to_local_H(local_H, x_new, user_specs, local_flag=1, on_cube=True)
                                new_inds_to_send_mgr.append(len(local_H) - 1)

                                run_order[child_idx].append(local_H[-1]["sim_id"])
                                run_pts[child_idx].append(x_new)
                                if local_H[-1]["sim_id"] in sim_id_to_child_inds:
                                    sim_id_to_child_inds[local_H[-1]["sim_id"]] += (child_idx,)
                                else:
                                    sim_id_to_child_inds[local_H[-1]["sim_id"]] = (child_idx,)

            starting_inds = decide_where_to_start_localopt(local_H, n, n_s, rk_const, ld, mu, nu)

            for ind in starting_inds:
                if len([p for p in local_opters.values() if p.is_running]) < user_specs.get("max_active_runs", np.inf):
                    local_H["started_run"][ind] = 1

                    # Initialize a local opt run
                    local_opter = LocalOptInterfacer(
                        user_specs,
                        local_H[ind]["x_on_cube"],
                        local_H[ind]["f"] if "f" in fields_to_pass else local_H[ind]["fvec"],
                        local_H[ind]["grad"] if "grad" in fields_to_pass else None,
                    )

                    local_opters[total_runs] = local_opter

                    x_new = local_opter.iterate(local_H[ind][fields_to_pass])  # Assuming the second x won't be optimal

                    add_to_local_H(local_H, x_new, user_specs, local_flag=1, on_cube=True)
                    new_inds_to_send_mgr.append(len(local_H) - 1)

                    run_order[total_runs] = [ind, local_H[-1]["sim_id"]]
                    run_pts[total_runs] = [local_H["x_on_cube"], x_new]

                    if local_H[-1]["sim_id"] in sim_id_to_child_inds:
                        sim_id_to_child_inds[local_H[-1]["sim_id"]] += (total_runs,)
                    else:
                        sim_id_to_child_inds[local_H[-1]["sim_id"]] = (total_runs,)

                    total_runs += 1

            if first_pass:
                num_samples = persis_info["nworkers"] - 1 - len(new_inds_to_send_mgr)
                first_pass = False
            else:
                num_samples = n_r - len(new_inds_to_send_mgr)

            if num_samples > 0:
                persis_info = add_k_sample_points_to_local_H(
                    num_samples, user_specs, persis_info, n, comm, local_H, sim_id_to_child_inds
                )
                new_inds_to_send_mgr = new_inds_to_send_mgr + list(range(len(local_H) - num_samples, len(local_H)))

            if not user_specs.get("standalone"):
                ps.send(local_H[new_inds_to_send_mgr + new_opt_inds_to_send_mgr][[i[0] for i in gen_specs["out"]]])
            something_sent = True

        return local_H, persis_info, FINISHED_PERSISTENT_GEN_TAG
    finally:
        try:
            clean_up_and_stop(local_opters)
        except NameError:
            pass


def update_local_H_after_receiving(local_H, n, n_s, user_specs, Work, calc_in, fields_to_pass):
    for name in ["f", "x_on_cube", "grad", "fvec"]:
        if name in fields_to_pass:
            assert name in calc_in.dtype.names, (
                name + " must be returned to persistent_aposmm for localopt_method: " + user_specs["localopt_method"]
            )

    for name in calc_in.dtype.names:
        local_H[name][Work["libE_info"]["H_rows"]] = calc_in[name]

    local_H["sim_ended"][Work["libE_info"]["H_rows"]] = True
    n_s += np.sum(~local_H[Work["libE_info"]["H_rows"]]["local_pt"])
    n_r = len(Work["libE_info"]["H_rows"])

    # dist -> distance
    update_history_dist(local_H, n)

    return n_s, n_r


def add_to_local_H(local_H, pts, user_specs, local_flag=0, on_cube=True):
    """
    Adds points to O, the numpy structured array to be sent back to the manager
    """
    assert not local_flag or len(pts) == 1, "Can't > 1 local points"

    len_local_H = len(local_H)

    ub = user_specs["ub"]
    lb = user_specs["lb"]

    num_pts = len(pts)

    local_H.resize(len(local_H) + num_pts, refcheck=False)  # Adds num_pts rows of zeros to O

    if on_cube:
        local_H["x_on_cube"][-num_pts:] = pts
        local_H["x"][-num_pts:] = pts * (ub - lb) + lb
    else:
        local_H["x_on_cube"][-num_pts:] = (pts - lb) / (ub - lb)
        local_H["x"][-num_pts:] = pts

    if user_specs.get("periodic"):
        local_H["x_on_cube"][-num_pts:] = local_H["x_on_cube"][-num_pts:] % 1

    local_H["sim_id"][-num_pts:] = np.arange(len_local_H, len_local_H + num_pts)
    local_H["local_pt"][-num_pts:] = local_flag

    initialize_dists_and_inds(local_H, num_pts)

    if local_flag:
        local_H["num_active_runs"][-num_pts] += 1


def initialize_dists_and_inds(local_H, num_pts):
    local_H["dist_to_unit_bounds"][-num_pts:] = np.inf
    local_H["dist_to_better_l"][-num_pts:] = np.inf
    local_H["dist_to_better_s"][-num_pts:] = np.inf
    local_H["ind_of_better_l"][-num_pts:] = -1
    local_H["ind_of_better_s"][-num_pts:] = -1


def update_history_dist(H, n):
    """
    Updates distances/indices after new points that have been evaluated.

    .. seealso::
        `start_persistent_local_opt_gens.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_persistent_local_opt_gens.py>`_
    """

    new_inds = np.where(~H["known_to_aposmm"])[0]

    p = np.logical_and.reduce((H["sim_ended"], ~np.isnan(H["f"])))

    for new_ind in new_inds:
        # Loop over new returned points and update their distances
        if p[new_ind]:
            H["known_to_aposmm"][new_ind] = True

            # Compute distance to boundary
            H["dist_to_unit_bounds"][new_ind] = min(
                min(np.ones(n) - H["x_on_cube"][new_ind]), min(H["x_on_cube"][new_ind] - np.zeros(n))
            )

            dist_to_all = cdist(H["x_on_cube"][[new_ind]], H["x_on_cube"][p], "euclidean").flatten()
            new_better_than = H["f"][new_ind] < H["f"][p]

            # Update any other points if new_ind is closer and better
            if H["local_pt"][new_ind]:
                inds_of_p = np.logical_and(dist_to_all < H["dist_to_better_l"][p], new_better_than)
                updates = np.where(p)[0][inds_of_p]
                H["dist_to_better_l"][updates] = dist_to_all[inds_of_p]
                H["ind_of_better_l"][updates] = new_ind
            else:
                inds_of_p = np.logical_and(dist_to_all < H["dist_to_better_s"][p], new_better_than)
                updates = np.where(p)[0][inds_of_p]
                H["dist_to_better_s"][updates] = dist_to_all[inds_of_p]
                H["ind_of_better_s"][updates] = new_ind

            # Since we allow equality when deciding better_than_new_l and
            # better_than_new_s, we have to prevent new_ind from being its own
            # better point.
            better_than_new_l = np.logical_and.reduce((~new_better_than, H["local_pt"][p], H["sim_id"][p] != new_ind))
            better_than_new_s = np.logical_and.reduce((~new_better_than, ~H["local_pt"][p], H["sim_id"][p] != new_ind))

            # Who is closest to ind and better
            if np.any(better_than_new_l):
                ind = dist_to_all[better_than_new_l].argmin()
                H["ind_of_better_l"][new_ind] = H["sim_id"][p][np.nonzero(better_than_new_l)[0][ind]]
                H["dist_to_better_l"][new_ind] = dist_to_all[better_than_new_l][ind]

            if np.any(better_than_new_s):
                ind = dist_to_all[better_than_new_s].argmin()
                H["ind_of_better_s"][new_ind] = H["sim_id"][p][np.nonzero(better_than_new_s)[0][ind]]
                H["dist_to_better_s"][new_ind] = dist_to_all[better_than_new_s][ind]

            # if not ignore_L8:
            #     r_k = calc_rk(len(H['x_on_cube'][0]), n_s, rk_const, lhs_divisions)
            #     H['worse_within_rk'][new_ind][p] = np.logical_and.reduce((H['f'][new_ind] <= H['f'][p], dist_to_all <= r_k))

            #     # Add trues if new point is 'worse_within_rk'
            #     inds_to_change = np.logical_and.reduce((H['dist_to_all'][p, new_ind] <= r_k, H['f'][new_ind] >= H['f'][p], H['sim_id'][p] != new_ind))
            #     H['worse_within_rk'][inds_to_change, new_ind] = True

            #     if not H['local_pt'][new_ind]:
            #         H['worse_within_rk'][H['dist_to_all'] > r_k] = False

    if np.any(~H["local_pt"]) and not np.any(np.isinf(H["dist_to_better_s"][~H["local_pt"]])):
        # Our best sample point was not identified because the min was not unique.
        min_inds = H["f"][~H["local_pt"]] == np.min(H["f"][~H["local_pt"]])
        assert len(min_inds) >= 2, "Check this"
        # Take the first point with this value to be the best sample point
        best_samp = H["sim_id"][~H["local_pt"]][min_inds][0]
        H["dist_to_better_s"][best_samp] = np.inf
        H["ind_of_better_s"][best_samp] = -1

    # if np.any(H['local_pt']) and not np.any(np.isinf(H['dist_to_better_l'][H['local_pt']])):
    #     # Our best sample point was not identified because the min was not unique.
    #     min_inds = H['f'][H['local_pt']] == np.min(H['f'][H['local_pt']])
    #     assert len(min_inds) >= 2, "Check this"
    #     # Take the first point with this value to be the best sample point
    #     best_local = H['sim_id'][H['local_pt']][min_inds][0]
    #     H['dist_to_better_l'][best_local] = np.inf
    #     H['ind_of_better_l'][best_local] = -1


def update_history_optimal(x_opt, opt_flag, H, run_inds):
    """
    Updated the history after any point has been declared a local minimum
    """

    # opt_ind = np.where(np.logical_and(np.equal(x_opt, H['x_on_cube']).all(1), ~np.isinf(H['f'])))[0] # This fails on some problems. x_opt is 1e-16 away from the point that was given and opt_ind is empty
    run_inds = np.unique(run_inds)

    dists = np.linalg.norm(H["x_on_cube"][run_inds] - x_opt, axis=1)
    ind = np.argmin(dists)
    opt_ind = run_inds[ind]

    tol_x1 = 1e-15

    # Instead of failing, we accept x_opt that is slightly different from its value in H
    # assert dists[ind] <= tol_x1, "Closest point to x_opt not within {}?".format(tol_x1)

    if dists[ind] > tol_x1:
        print(
            "[APOSMM] Dist from reported x_opt to closest evaluated point is: "
            + str(dists[ind])
            + "\n"
            + "[APOSMM] Check that the local optimizer is working correctly\n",
            x_opt,
            run_inds,
            flush=True,
        )

    tol_x2 = 1e-8
    failsafe = np.logical_and(H["f"][run_inds] < H["f"][opt_ind], dists < tol_x2)
    if opt_flag:
        if np.any(failsafe):
            print(
                f"[APOSMM] This run has {sum(failsafe)} point(s) with smaller 'f' value within {tol_x2} of "
                "the point ruled to be the run minimum. \nMarking all as being "
                "a 'local_min' to prevent APOSMM from starting another run "
                "immediately from these points."
            )
            print("[APOSMM] Sim_ids to be marked optimal: ", opt_ind, run_inds[failsafe])
            print("[APOSMM] Check that the local optimizer is working correctly", flush=True)
            H["local_min"][run_inds[failsafe]] = 1

        H["local_min"][opt_ind] = 1

    H["num_active_runs"][run_inds] -= 1

    return opt_ind


def decide_where_to_start_localopt(H, n, n_s, rk_const, ld=0, mu=0, nu=0):
    """
    APOSMM starts a local optimization runs from a point that:

    - is not in an active local optimization run,
    - is more than ``mu`` from the boundary (in the unit-cube domain),
    - is more than ``nu`` from identified minima (in the unit-cube domain),
    - does not have a better point within a distance ``r_k`` of it.


    For further details, see the conditions (S1-S5 and L1-L8) in Table 1 of the
    `APOSMM paper <https://doi.org/10.1007/s12532-017-0131-4>`_
    This method first identifies sample points satisfying S2-S5, and then
    identifies all localopt points that satisfy L1-L7.
    We then start from any sample point also satisfying S1.
    We do not check condition L8 currently.

    We don't consider points in the history that have not returned from
    computation, or that have a ``nan`` value. As APOSMM works on the unit
    cube, note that ``mu`` and ``nu`` implicitly depend on the scaling of the
    original domain: adjusting the initial domain can make a run start (or not
    start) at a point that didn't (or did) previously.

    Parameters
    ----------
    H: numpy.ndarray
        History array storing rows for each point. Numpy structured array.
    n: int
        Problem dimension
    n_s: int
        Number of sample points in H
    r_k_const: float
        Radius for deciding when to start runs
    ld: int
        Number of Latin hypercube sampling divisions (0 or 1 means uniform
        random sampling over the domain)
    mu: float
        Nonnegative distance from the boundary that all starting points must satisfy
    nu: float
        Nonnegative distance from identified minima that all starting points must satisfy

    Returns
    ----------
    start_inds: list
        Indices where a local opt run should be started, sorted by increasing
        function value.


    .. seealso::
        `start_persistent_local_opt_gens.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_persistent_local_opt_gens.py>`_
    """

    r_k = calc_rk(n, n_s, rk_const, ld)

    if nu > 0:
        test_2_through_5 = np.logical_and.reduce(
            (
                H["sim_ended"] == 1,  # have a returned function value
                H["dist_to_better_s"] > r_k,  # no better sample point within r_k (L2)
                ~H["started_run"],  # have not started a run (L3)
                H["dist_to_unit_bounds"] >= mu,  # have all components at least mu away from bounds (L4)
                np.all(
                    cdist(H["x_on_cube"], H["x_on_cube"][H["local_min"]]) >= nu, axis=1
                ),  # distance nu away from known local mins (L5)
            )
        )
    else:
        test_2_through_5 = np.logical_and.reduce(
            (
                H["sim_ended"] == 1,  # have a returned function value
                H["dist_to_better_s"] > r_k,  # no better sample point within r_k (L2)
                ~H["started_run"],  # have not started a run (L3)
                H["dist_to_unit_bounds"] >= mu,  # have all components at least mu away from bounds (L4)
            )
        )  # (L5) is always true when nu = 0

    # assert gamma_quantile == 1, "This is not supported yet. What is the best way to decide this when there are NaNs present in H['f']?"
    # if gamma_quantile < 1:
    #     cut_off_value = np.sort(H['f'][~H['local_pt']])[np.floor(gamma_quantile*(sum(~H['local_pt'])-1)).astype(int)]
    # else:
    #     cut_off_value = np.inf

    # Find the indices of points that...
    sample_seeds = np.logical_and.reduce(
        (
            ~H["local_pt"],  # are not localopt points
            # H['f'] <= cut_off_value,      # have a small enough objective value
            ~np.isinf(H["f"]),  # have a non-infinity objective value
            ~np.isnan(H["f"]),  # have a non-NaN objective value
            test_2_through_5,  # satisfy tests 2 through 5
        )
    )

    # Uncomment the following to test the effect of ignoring LocalOpt points
    # in APOSMM. This allows us to test a parallel MLSL.
    # return list(np.ix_(sample_seeds)[0])

    those_satisfying_S1 = H["dist_to_better_l"][sample_seeds] > r_k  # no better localopt point within r_k
    sample_start_inds = np.ix_(sample_seeds)[0][those_satisfying_S1]

    # Find the indices of points that...
    local_seeds = np.logical_and.reduce(
        (
            H["local_pt"],  # are localopt points
            H["dist_to_better_l"] > r_k,  # no better local point within r_k (L1)
            ~np.isinf(H["f"]),  # have a non-infinity objective value
            ~np.isnan(H["f"]),  # have a non-NaN objective value
            test_2_through_5,
            H["num_active_runs"] == 0,  # are not in an active run (L6)
            ~H["local_min"],  # are not a local min (L7)
        )
    )

    local_start_inds2 = list(np.ix_(local_seeds)[0])

    # If paused is a field in H, don't start from paused points.
    if "paused" in H.dtype.names:
        sample_start_inds = sample_start_inds[~H[sample_start_inds]["paused"]]
    start_inds = list(sample_start_inds) + local_start_inds2

    # Sort the starting inds by their function value
    inds = np.argsort(H["f"][start_inds])
    start_inds = np.array(start_inds)[inds].tolist()

    return start_inds


def calc_rk(n, n_s, rk_const, lhs_divisions=0):
    """Calculate the critical distance r_k"""
    if lhs_divisions == 0:
        if n_s == 1:
            r_k = 1e8
        else:
            r_k = rk_const * (log(n_s) / n_s) ** (1 / n)
    else:
        k = np.floor(n_s / lhs_divisions).astype(int)
        if k <= 1:  # to prevent r_k=0
            r_k = np.inf
        else:
            r_k = rk_const * (log(k) / k) ** (1 / n)

    return r_k


def extract_rk_c(user_specs, n):
    return user_specs.get("rk_const", ((gamma(1 + (n / 2.0)) * 5.0) ** (1.0 / n)) / sqrt(pi))


def initialize_APOSMM(H, user_specs, libE_info):
    """
    Computes common values every time that APOSMM is reinvoked

    .. seealso::
        `start_persistent_local_opt_gens.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/alloc_funcs/start_persistent_local_opt_gens.py>`_
    """
    n = len(user_specs["ub"])

    rk_c = extract_rk_c(user_specs, n)
    ld = user_specs.get("lhs_divisions", 0)
    mu = user_specs.get("mu", 1e-4)
    nu = user_specs.get("nu", 0)

    comm = libE_info["comm"] if not user_specs.get("standalone") else []

    local_H_fields = [
        ("f", float),
        ("grad", float, n),
        ("x", float, n),
        ("x_on_cube", float, n),
        ("local_pt", bool),
        ("known_to_aposmm", bool),
        ("dist_to_unit_bounds", float),
        ("dist_to_better_l", float),
        ("dist_to_better_s", float),
        ("ind_of_better_l", int),
        ("ind_of_better_s", int),
        ("started_run", bool),
        ("num_active_runs", int),
        ("local_min", bool),
        ("sim_id", int),
        ("paused", bool),
        ("sim_ended", bool),
    ]

    if "components" in user_specs:
        local_H_fields += [("fvec", float, user_specs["components"])]

    local_H = np.zeros(len(H), dtype=local_H_fields)

    if len(H):
        for field in H.dtype.names:
            local_H[field][: len(H)] = H[field]

        if user_specs["localopt_method"] in ["LD_MMA", "blmvm"]:
            assert "grad" in H.dtype.names, (
                "Must give 'grad' values to persistent_aposmm in gen_specs['in'] when using 'localopt_method'"
                + user_specs["localopt_method"]
            )
            assert not np.all(local_H["grad"] == 0), "All 'grad' values are zero for the given points."

        assert "f" in H.dtype.names, "Must give 'f' values to persistent_aposmm in gen_specs['in']"
        assert "sim_id" in H.dtype.names, "Must give 'sim_id' to persistent_aposmm in gen_specs['in']"
        assert "sim_ended" in H.dtype.names, "Must give 'sim_ended' status to persistent_aposmm in gen_specs['in']"

        over_written_fields = [
            "dist_to_unit_bounds",
            "dist_to_better_l",
            "dist_to_better_s",
            "ind_of_better_l",
            "ind_of_better_s",
        ]
        if any([i in H.dtype.names for i in over_written_fields]):
            print("\n[APOSMM] Ignoring given values in these fields: " + str(over_written_fields) + "\n")

        initialize_dists_and_inds(local_H, len(H))

        # Update after receiving initial points
        update_history_dist(local_H, n)

    n_s = np.sum(~local_H["local_pt"])

    msg = "APOSMM requires a positive initial_sample_size, or some existing points in order to determine where to start local optimization runs."
    assert n_s > 0 or user_specs["initial_sample_size"] > 0, msg

    if "sample_points" in user_specs:
        assert user_specs["sample_points"].ndim == 2, "Must have 2 dimensions for sample points"
        assert isinstance(user_specs["sample_points"], np.ndarray)

    return n, n_s, rk_c, ld, mu, nu, comm, local_H


def initialize_children(user_specs):
    """Initialize stuff for localopt children"""
    local_opters = {}
    sim_id_to_child_inds = {}
    run_order = {}
    run_pts = {}  # These can differ from 'x_on_cube' (e.g., if user_specs['periodic']=1 and runs leave unit cube)
    total_runs = 0
    ended_runs = []
    if user_specs["localopt_method"] in ["LD_MMA", "blmvm", "scipy_BFGS"]:
        fields_to_pass = ["x_on_cube", "f", "grad"]
    elif user_specs["localopt_method"] in [
        "LN_SBPLX",
        "LN_BOBYQA",
        "LN_COBYLA",
        "LN_NEWUOA",
        "LN_NELDERMEAD",
        "scipy_Nelder-Mead",
        "scipy_COBYLA",
        "external_localopt",
        "nm",
    ]:
        fields_to_pass = ["x_on_cube", "f"]
    elif user_specs["localopt_method"] in ["pounders", "ibcdfo_pounders", "ibcdfo_manifold_sampling", "dfols"]:
        fields_to_pass = ["x_on_cube", "fvec"]
    else:
        raise NotImplementedError(f"Unknown local optimization method {user_specs['localopt_method']}.")

    return local_opters, sim_id_to_child_inds, run_order, run_pts, total_runs, ended_runs, fields_to_pass


def add_k_sample_points_to_local_H(k, user_specs, persis_info, n, comm, local_H, sim_id_to_child_inds):
    if "sample_points" in user_specs:
        v = np.sum(~local_H["local_pt"])  # Number of sample points so far
        sampled_points = user_specs["sample_points"][v : v + k]
        on_cube = False  # Assume points are on original domain, not unit cube
        if len(sampled_points):
            add_to_local_H(local_H, sampled_points, user_specs, on_cube=on_cube)
        k = k - len(sampled_points)

    if k > 0:
        sampled_points = persis_info["rand_stream"].uniform(0, 1, (k, n))
        add_to_local_H(local_H, sampled_points, user_specs, on_cube=True)

    return persis_info


# def clean_up_and_stop(local_H, local_opters):
def clean_up_and_stop(local_opters):
    # FIXME: This has to be a clean exit.

    # print('[Parent]: The optimal points and values are:\n',
    #       local_H[np.where(local_H['local_min'])][['x', 'f']], flush=True)

    for i, p in local_opters.items():
        p.destroy()


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
