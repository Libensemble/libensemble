"""
@Author: Created by Caleb Ju at Argonne National Labs as Given's associate,
         Summer 2021.
"""

import numpy as np
import scipy.sparse as spp

from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def start_consensus_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):
    """
    Many distributed optimization algorithms require two non-local, (e.g., not
    elementwise addition or multiplication) operations: evaluate gradients and
    consensus, which is just a linear combination of a nodes's neighbors'
    solutions. Thus, we develop a general alloc function that does both of
    these.

    From the caller function, the user must pass a square matrix in
    `persis_info["A"]. Typically, this is a Laplacian matrix of a connected
    graph or a doubly stochastic matrix. Note that the alloc will send the
    row i's non-zero indices values to gen i when initializing the gen
    via the persis_info object. This information is used methods such as
    `gen_funcs/persistent_n_agent.py`, where each gen needs to take
    a linear combination of its neighbors's `x`'s.

    The user has three remaining optional values to pass into the alloc,
    which are all set from the calling script.

    - 1. The user can set `persis_info["print_progress"]=1`
    This tells alloc function print the consensus value and iteration count
    whenever a consensus step is taken

    - 2. The user can set `persis_info["gen_params"]={... dictionary ...}`.
    The dictionary contains the any parameters that the gen will utilize.
    This can include, for example, functions that the gen can use to compute
    gradients rather than requesting a sim to complete the work (see the
    linear regression tests in `tests/regression_tests/test_persistent_pds.py`)

    - 3. The user can set `persis_info["sim_params"]={... dictionary ...}`.
    Similarly, the dictionary contains the any parameters that the sim will
    utilize

                ------------------------------------------------

    Now, we briefly explain how to request the gradient of the consensus term
    or the gradient of the function as the gen.

    For the former, the gen submits a work request and must set the parameter
    `consensus_pt` to True, while making sure to pass into the `x` variable.

    If the user wants the alloc to sum all the {f_i}, the user must set both
    `consensus_pt` and `eval_pt` to True while settings the `f_i` variable. (The
    reason for having both variables set to True is to simplify the implementation
    in the alloc.)

    Finally, to request a gradient of f_i, `consensus_pt` must be set to False,
    `get_grad` needs to be set to True, `obj_component` (i.e., which f_i to
    consider) must be set, and the `x` variable must be set.

    If the user wants a function evaluation, then set `get_grad` to False instead.
    """

    if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
        return {}, persis_info

    # Initialize alloc_specs["user"] as user.
    user = alloc_specs.get("user", {})
    manage_resources = "resource_sets" in H.dtype.names or libE_info["use_resource_sets"]
    support = AllocSupport(W, manage_resources, persis_info, libE_info)
    gen_count = support.count_persis_gens()
    Work = {}
    is_first_iter = False

    if persis_info.get("first_call", True):
        persis_info.update({"last_H_len": 0})
        persis_info.update({"next_to_give": 0})
        persis_info.update({"first_call": False})
        is_first_iter = True

        A = persis_info.get("A")
        assert A.shape[0] == A.shape[1], "Matrix @A is not square"

    # Exit if all persistent gens are done
    elif gen_count == 0:
        return Work, persis_info, 1

    num_gens_at_consensus = 0

    # Sort to get consistent ordering for processing workers
    avail_persis_worker_ids = np.sort(support.avail_worker_ids(persistent=EVAL_GEN_TAG))

    for wid in avail_persis_worker_ids:
        # If at consensus, wait until everyone is done
        if persis_info[wid].get("at_consensus", False):
            num_gens_at_consensus += 1
            continue

        # Gen is waiting on sim work
        elif len(persis_info[wid].get("curr_H_ids", [])):
            [l_H_id, r_H_id] = persis_info[wid].get("curr_H_ids")
            num_sims_req = r_H_id - l_H_id

            num_fin_sims = np.sum(H["sim_ended"][l_H_id:r_H_id])

            completed_all_sims_for_gen_i = num_fin_sims == num_sims_req

            # if completed all work, send back
            if completed_all_sims_for_gen_i:
                sims_to_ret_to_gen = np.arange(l_H_id, r_H_id)

                Work[wid] = support.gen_work(
                    wid,
                    ["x", "f_i", "gradf_i"],
                    sims_to_ret_to_gen,
                    persis_info.get(wid),
                    persistent=True,
                )

                persis_info[wid].update({"curr_H_ids": []})

        # otherwise, check if gen has requested new work
        else:
            last_H_len = persis_info["last_H_len"]

            # did gen sent consensus? (start @last_H_len to avoid old work)
            consensus_sim_ids = np.where(
                np.logical_and(H[last_H_len:]["consensus_pt"], H[last_H_len:]["gen_worker"] == wid)
            )[0]

            if len(consensus_sim_ids) > 0:
                assert len(consensus_sim_ids) == 1, (
                    "Gen should only send one " + f"point for consensus step, received {len(consensus_sim_ids)}"
                )

                # re-center (since the last_H_len has relative index 0)
                sim_id = consensus_sim_ids[0] + last_H_len

                persis_info[wid].update({"curr_H_ids": [sim_id, sim_id + 1]})
                persis_info[wid].update({"at_consensus": True})

                num_gens_at_consensus += 1

            # otherwise, gen requested new work for sim
            else:
                new_H_ids_from_gen_i = np.where(H[last_H_len:]["gen_worker"] == wid)[0]

                assert len(new_H_ids_from_gen_i) > 0, (
                    "Gen must request new sim " + "work or show convergence if avail, but neither occurred"
                )

                new_H_ids_from_gen_i += last_H_len

                l_H_id = new_H_ids_from_gen_i[0]
                r_H_id = new_H_ids_from_gen_i[-1] + 1

                # (!!)
                assert len(new_H_ids_from_gen_i) == r_H_id - l_H_id, "new gen " + "data must be in contiguous space"

                persis_info[wid].update({"curr_H_ids": [l_H_id, r_H_id]})

    # If all gens at consensus, distribute data to all gens
    if num_gens_at_consensus == user["num_gens"]:
        assert num_gens_at_consensus == len(
            avail_persis_worker_ids
        ), f"All gens must be available, only {len(avail_persis_worker_ids)}/{len(num_gens_at_consensus)} are though..."

        # get index in history array @H where each gen's consensus point lies
        consensus_ids_in_H = np.array([persis_info[i]["curr_H_ids"][0] for i in avail_persis_worker_ids], dtype=int)

        # Setup for printing progress
        print_progress = persis_info.get("print_progress", False)
        print_obj = H[consensus_ids_in_H[0]]["eval_pt"]
        fsum = 0

        if print_progress:
            num_gens = user["num_gens"]
            n = len(gen_specs["user"]["lb"])
            Ax = np.empty(num_gens * n, dtype=float)
            x = np.empty(num_gens * n, dtype=float)
            # if (1st) gen asks to gather all f_i's and print their sum

        A = persis_info["A"]
        for i0, wid in enumerate(avail_persis_worker_ids):
            incident_gens = A.indices[A.indptr[i0] : A.indptr[i0 + 1]]
            # remove own index
            own_idx = np.argwhere(incident_gens == i0)
            incident_gens = np.delete(incident_gens, own_idx)

            neighbor_consensus_ids_in_H = consensus_ids_in_H[incident_gens]

            if print_progress:
                # implicitly perform matmul, $(A \kron I)[x_1,...x_m]$
                x[i0 * n : (i0 + 1) * n] = H[consensus_ids_in_H[i0]]["x"]

                diag_scalar = A.diagonal()[i0]
                diag_term = diag_scalar * H[consensus_ids_in_H[i0]]["x"]

                offdiag_scalars = A.data[A.indptr[i0] : A.indptr[i0 + 1]]
                offdiag_scalars = np.delete(offdiag_scalars, own_idx)
                offdiag_terms = spp.diags(offdiag_scalars).dot(H[neighbor_consensus_ids_in_H]["x"])
                offdiag_term = np.sum(offdiag_terms, axis=0)

                Ax[i0 * n : (i0 + 1) * n] = diag_term + offdiag_term

            if print_obj:
                fsum += H[consensus_ids_in_H[i0]]["f_i"]

            Work[wid] = support.gen_work(
                wid,
                ["x", "gen_worker"],
                np.atleast_1d(neighbor_consensus_ids_in_H),
                persis_info.get(wid),
                persistent=True,
            )

            persis_info[wid].update({"curr_H_ids": []})
            persis_info[wid].update({"at_consensus": False})

        if print_obj and print_progress:
            msg = f"F(x)={fsum:.8f}\n"
            print(f"{msg}con={np.dot(x, Ax):.4e}", flush=True)
        elif print_obj:
            print(f"F(x)={fsum:.8f}", flush=True)
        elif print_progress:
            print(f"con={np.dot(x, Ax):.4e}", flush=True)

    # partition sum of convex functions evenly (only do at beginning)
    if is_first_iter and len(support.avail_worker_ids(persistent=False)):
        num_funcs = user["m"]
        num_gens = user["num_gens"]
        num_funcs_arr = partition_funcs_arr(num_funcs, num_gens)

    inactive_workers = np.sort(support.avail_worker_ids(persistent=False))
    for i0, wid in enumerate(inactive_workers):
        # start up gens
        if is_first_iter and gen_count < user["num_gens"]:
            # Checking resources first before call to gen_work
            rset_team = None
            if support.manage_resources:
                gen_resources = support.persis_info.get("gen_resources", 0)
                try:
                    rset_team = support.assign_resources(gen_resources)
                except InsufficientFreeResources:
                    break

            A = persis_info["A"]

            gen_count += 1
            l_idx = num_funcs_arr[gen_count - 1]
            r_idx = num_funcs_arr[gen_count]

            A_i_indices = A.indices[A.indptr[i0] : A.indptr[i0 + 1]]
            A_i_gen_ids = inactive_workers[A_i_indices]
            # gen A_i_gen_ids[wid] corresponds to weight S_i_data[wid]
            A_i_data = A.data[A.indptr[i0] : A.indptr[i0 + 1]]

            persis_info[wid].update(
                {
                    "f_i_idxs": range(l_idx, r_idx),
                    "A_i_gen_ids": A_i_gen_ids,
                    "A_i_data": A_i_data,
                    "params": persis_info.get("gen_params", {}),
                }
            )
            persis_info[wid].update({"at_consensus": False, "curr_H_ids": []})

            Work[wid] = support.gen_work(
                wid, gen_specs.get("in", []), range(len(H)), persis_info.get(wid), persistent=True, rset_team=rset_team
            )

        # give sim work when task available
        elif persis_info["next_to_give"] < len(H):
            # skip points that are not sim work or are already done
            while persis_info["next_to_give"] < len(H) and (
                H[persis_info["next_to_give"]]["sim_started"]
                or H[persis_info["next_to_give"]]["consensus_pt"]
                or H[persis_info["next_to_give"]]["cancel_requested"]
            ):
                persis_info["next_to_give"] += 1

            if persis_info["next_to_give"] >= len(H):
                break

            # Checking resources first before call to sim_work
            rset_team = None
            if support.manage_resources:
                num_rsets_req = np.max(H[persis_info["next_to_give"]]["resource_sets"])
                try:
                    rset_team = support.assign_resources(num_rsets_req)
                except InsufficientFreeResources:
                    break

            gen_id = H[persis_info["next_to_give"]]["gen_worker"]
            [l_H_ids, r_H_ids] = persis_info[gen_id]["curr_H_ids"]

            assert (
                l_H_ids == persis_info["next_to_give"]
            ), f"@next_to_give={persis_info['next_to_give']} does not match gen's requested work H id of {l_H_ids}"

            persis_info[wid].update({"params": persis_info.get("sim_params", {})})

            Work[wid] = support.sim_work(
                wid, H, sim_specs["in"], np.arange(l_H_ids, r_H_ids), persis_info.get(wid), rset_team=rset_team
            )

            # we can safely assume the rows are contiguous due to (!!)
            persis_info["next_to_give"] += r_H_ids - l_H_ids

        else:
            break

    persis_info.update({"last_H_len": len(H)})

    return Work, persis_info, 0


def partition_funcs_arr(num_funcs, num_gens):
    """This evenly divides the functions amongst the gens. For instance,
        when there are say 7 functions and 4 gens, this function will
        distribute 2 contiguous functions to the first 3 gens, and then
        function to the last gen.

    Parameters
    ----------
    - num_funcs : int
        How {f_i}'s there are
    - num_gens : int
        How many gens

    Returns
    -------
    - num_funcs_arr : np.ndarray
        Index pointer (i.e. gen i has functions [arr[i],arr[i+1])
    """
    num_funcs_arr = (num_funcs // num_gens) * np.ones(num_gens, dtype=int)
    num_leftover_funcs = num_funcs % num_gens
    num_funcs_arr[:num_leftover_funcs] += 1

    # builds starting and ending function indices for each gen e.g. if 7
    # functions split up amongst 3 gens, then num_funcs__arr = [0, 3, 5, 7]
    num_funcs_arr = np.append(0, np.cumsum(num_funcs_arr))

    return num_funcs_arr
