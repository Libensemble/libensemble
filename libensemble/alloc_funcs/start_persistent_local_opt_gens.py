import numpy as np
from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources
from libensemble.gen_funcs.old_aposmm import initialize_APOSMM, decide_where_to_start_localopt, update_history_dist


def start_persistent_local_opt_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):
    """
    This allocation function will do the following:

    - Start up a persistent generator that is a local opt run at the first point
      identified by APOSMM's decide_where_to_start_localopt. Note, it will do
      this only if at least one worker will be left to perform simulation evaluations.
    - If multiple starting points are available, the one with smallest function
      value is chosen.
    - If no candidate starting points exist, points from existing runs will be
      evaluated (oldest first).
    - If no points are left, call the generation function.

    tags: alloc, persistent, aposmm

    .. seealso::
        `test_uniform_sampling_then_persistent_localopt_runs.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_then_persistent_localopt_runs.py>`_ # noqa
    """

    if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
        return {}, persis_info

    manage_resources = "resource_sets" in H.dtype.names or libE_info["use_resource_sets"]
    support = AllocSupport(W, manage_resources, persis_info, libE_info)
    Work = {}
    gen_count = support.count_persis_gens()
    points_to_evaluate = ~H["sim_started"] & ~H["cancel_requested"]

    # If a persistent localopt run has just finished, use run_order to update H
    # and then remove other information from persis_info
    for i in persis_info.keys():
        if "done" in persis_info[i]:
            H["num_active_runs"][persis_info[i]["run_order"]] -= 1
        if "x_opt" in persis_info[i]:
            opt_ind = np.all(H["x"] == persis_info[i]["x_opt"], axis=1)
            # assert sum(opt_ind) == 1, "There must be just one optimum"
            H["local_min"][opt_ind] = True
            persis_info[i] = {"rand_stream": persis_info[i]["rand_stream"]}
            return Work, persis_info, 1 # Stop after one run finishes

    # If wid is idle, but in persistent mode, and its calculated values have
    # returned, give them back to i. Otherwise, give nothing to wid
    for wid in support.avail_worker_ids(persistent=EVAL_GEN_TAG):
        gen_inds = H["gen_worker"] == wid
        if support.all_sim_ended(H, gen_inds):
            last_time_pos = np.argmax(H["sim_started_time"][gen_inds])
            last_ind = np.nonzero(gen_inds)[0][last_time_pos]
            Work[wid] = support.gen_work(wid, gen_specs["persis_in"], last_ind, persis_info[wid], persistent=True)
            persis_info[wid]["run_order"].append(last_ind)

    for wid in support.avail_worker_ids(persistent=False):
        # Find candidates to start local opt runs if a sample has been evaluated
        if np.any(np.logical_and(~H["local_pt"], H["sim_ended"], ~H["cancel_requested"])):
            n, _, _, _, r_k, mu, nu = initialize_APOSMM(H, gen_specs)
            update_history_dist(H, n, gen_specs["user"], c_flag=False)
            starting_inds = decide_where_to_start_localopt(H, r_k, mu, nu)
        else:
            starting_inds = []

        # Start persistent generator for local opt run unless it would use all workers
        if starting_inds and gen_count + 1 < len(W):
            # Start at the best possible starting point
            ind = starting_inds[np.argmin(H["f"][starting_inds])]
            try:
                Work[wid] = support.gen_work(wid, gen_specs["persis_in"], ind, persis_info[wid], persistent=True)
            except InsufficientFreeResources:
                break
            H["started_run"][ind] = 1
            H["num_active_runs"][ind] += 1
            persis_info[wid]["run_order"] = [ind]
            gen_count += 1

        elif np.any(points_to_evaluate):

            # Perform sim evaluations from existing runs
            q_inds_logical = np.logical_and(points_to_evaluate, H["local_pt"])
            if not np.any(q_inds_logical):
                q_inds_logical = points_to_evaluate
            sim_ids_to_send = np.nonzero(q_inds_logical)[0][0]  # oldest point
            try:
                Work[wid] = support.sim_work(wid, H, sim_specs["in"], sim_ids_to_send, [])
            except InsufficientFreeResources:
                break
            points_to_evaluate[sim_ids_to_send] = False

        elif gen_count == 0 and not np.any(np.logical_and(W["active"] == EVAL_GEN_TAG, W["persis_state"] == 0)):
            # Finally, generate points since there is nothing else to do (no resource sets req.)
            Work[wid] = support.gen_work(wid, gen_specs.get("in", []), [], persis_info[wid], rset_team=[])
            gen_count += 1

    return Work, persis_info
