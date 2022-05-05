import numpy as np
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If all entries in `H` have been given a be evaluated, a worker
    is told to call the generator function, provided this wouldn't result in
    more than ``alloc_specs['user']['num_active_gen']`` active generators. Also allows
    for a 'batch_mode'.

    When there are multiple objective components, this allocation function
    does not evaluate further components for some point in the following
    scenarios:

    alloc_specs['user']['stop_on_NaNs']: True --- after a NaN has been found in returned in some
        objective component
    alloc_specs['user']['stop_partial_fvec_eval']: True --- after the value returned from
        combine_component_func is larger than a known upper bound on the objective.

    .. seealso::
        `test_uniform_sampling_one_residual_at_a_time.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_one_residual_at_a_time.py>`_ # noqa
    """

    if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
        return {}, persis_info

    manage_resources = "resource_sets" in H.dtype.names or libE_info["use_resource_sets"]
    support = AllocSupport(W, manage_resources, persis_info, libE_info)
    Work = {}
    gen_count = support.count_gens()

    if gen_specs["user"].get("single_component_at_a_time"):
        assert alloc_specs["user"]["batch_mode"], "Must be in batch mode when using 'single_component_at_a_time'"
    if len(H) != persis_info["H_len"]:
        # Something new is in the history.
        persis_info["need_to_give"].update(H["sim_id"][persis_info["H_len"] :].tolist())
        persis_info["H_len"] = len(H)
        persis_info["pt_ids"] = set(np.unique(H["pt_id"]))
        for pt_id in persis_info["pt_ids"]:
            persis_info["inds_of_pt_ids"][pt_id] = H["pt_id"] == pt_id

    idle_workers = support.avail_worker_ids()

    while len(idle_workers):

        pt_ids_to_pause = set()

        # Find indices of H that are not yet given out to be evaluated
        if len(persis_info["need_to_give"]):
            # If 'stop_on_NaN' is true and any f_i is a NaN, then pause
            # evaluations of other f_i, corresponding to the same pt_id
            if alloc_specs["user"].get("stop_on_NaNs"):
                pt_ids_to_pause.update(H["pt_id"][np.isnan(H["f_i"])])

            # If 'stop_partial_fvec_eval' is true, pause entries in H if a
            # partial combine_component_func evaluation is # worse than the
            # best, known, complete evaluation (and the point is not a
            # local_pt).
            if alloc_specs["user"].get("stop_partial_fvec_eval"):
                pt_ids = set(persis_info["pt_ids"]) - persis_info["has_nan"] - persis_info["complete"]
                pt_ids = np.array(list(pt_ids))
                partial_fvals = np.zeros(len(pt_ids))

                # Mark 'complete' and 'has_nan' pt_ids, compute complete and partial fvals
                for j, pt_id in enumerate(pt_ids):

                    a1 = persis_info["inds_of_pt_ids"][pt_id]
                    if np.any(np.isnan(H["f_i"][a1])):
                        persis_info["has_nan"].add(pt_id)
                        continue

                    if "local_pt" in H.dtype.names and H["local_pt"][a1][0]:
                        persis_info["local_pt_ids"].add(pt_id)

                    if np.all(H["sim_ended"][a1]):
                        persis_info["complete"].add(pt_id)
                        values = gen_specs["user"]["combine_component_func"](H["f_i"][a1])
                        persis_info["best_complete_val"] = min(persis_info["best_complete_val"], values)
                    else:
                        # Ensure combine_component_func calculates partial fevals correctly
                        # with H['f_i'] = 0 for non-returned point
                        partial_fvals[j] = gen_specs["user"]["combine_component_func"](H["f_i"][a1])

                if len(persis_info["complete"]) and len(pt_ids) > 1:

                    worse_flag = np.zeros(len(pt_ids), dtype=bool)
                    for j, pt_id in enumerate((pt_ids)):
                        if (
                            (not np.isnan(partial_fvals[j]))
                            and (pt_id not in persis_info["local_pt_ids"])
                            and (pt_id not in persis_info["complete"])
                            and (partial_fvals[j] > persis_info["best_complete_val"])
                        ):
                            worse_flag[j] = True

                    # Pause incomplete evaluations with worse_flag==True
                    pt_ids_to_pause.update(pt_ids[worse_flag])

            if not pt_ids_to_pause.issubset(persis_info["already_paused"]):
                persis_info["already_paused"].update(pt_ids_to_pause)
                sim_ids_to_remove = np.in1d(H["pt_id"], list(pt_ids_to_pause))
                H["paused"][sim_ids_to_remove] = True

                persis_info["need_to_give"] = persis_info["need_to_give"].difference(np.where(sim_ids_to_remove)[0])

            if len(persis_info["need_to_give"]) != 0:
                next_row = persis_info["need_to_give"].pop()
                i = idle_workers[0]
                try:
                    Work[i] = support.sim_work(i, H, sim_specs["in"], [next_row], [])
                except InsufficientFreeResources:
                    persis_info["need_to_give"].add(next_row)
                    break
                idle_workers = idle_workers[1:]

        elif gen_count < alloc_specs["user"].get("num_active_gens", gen_count + 1):
            lw = persis_info["last_worker"]

            last_size = persis_info.get("last_size")
            if len(H):
                # Don't give gen instances in batch mode if points are unfinished
                if alloc_specs["user"].get("batch_mode") and not all(
                    np.logical_or(H["sim_ended"][last_size:], H["paused"][last_size:])
                ):
                    break
                # Don't call APOSMM if there are runs going but none need advancing
                if len(persis_info[lw]["run_order"]):
                    runs_needing_to_advance = np.zeros(len(persis_info[lw]["run_order"]), dtype=bool)
                    for run, inds in enumerate(persis_info[lw]["run_order"].values()):
                        runs_needing_to_advance[run] = np.all(H["sim_ended"][inds])

                    if not np.any(runs_needing_to_advance):
                        break

            # Give gen work
            i = idle_workers[0]
            try:
                Work[i] = support.gen_work(i, gen_specs["in"], range(len(H)), persis_info[lw])
            except InsufficientFreeResources:
                break
            idle_workers = idle_workers[1:]
            gen_count += 1
            persis_info["total_gen_calls"] += 1
            persis_info["last_worker"] = i
            persis_info["last_size"] = len(H)

        elif gen_count >= alloc_specs["user"].get("num_active_gens", gen_count + 1):
            idle_workers = []

    return Work, persis_info
