import numpy as np
from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def persistent_aposmm_alloc(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start a persistent APOSMM generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    This function assumes that one persistent APOSMM will be started and never
    stopped (until some exit_criterion is satisfied).

    .. seealso::
        `test_persistent_aposmm_with_grad.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_with_grad.py>`_ # noqa
    """

    if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
        return {}, persis_info

    init_sample_size = gen_specs["user"]["initial_sample_size"]
    manage_resources = "resource_sets" in H.dtype.names or libE_info["use_resource_sets"]
    support = AllocSupport(W, manage_resources, persis_info, libE_info)
    gen_count = support.count_persis_gens()
    Work = {}

    if persis_info.get("first_call", True):
        assert support.all_sim_started(H), "Initial points in H have never been given."
        assert support.all_sim_ended(H), "Initial points in H have never been returned."
        assert support.all_gen_informed(H), "Initial points in H have never been given back to gen."

        persis_info["samples_in_H0"] = sum(H["local_pt"] == 0)
        persis_info["next_to_give"] = len(H)  #
        persis_info["first_call"] = False
    elif gen_count == 0:
        # The one persistent gen is done. Exiting
        return Work, persis_info, 1

    # If any persistent worker's calculated values have returned, give them back.
    for wid in support.avail_worker_ids(persistent=EVAL_GEN_TAG):
        if persis_info.get("sample_done") or sum(H["sim_ended"]) >= init_sample_size + persis_info["samples_in_H0"]:
            # Don't return if the initial sample is not complete
            persis_info["sample_done"] = True

            returned_but_not_given = np.logical_and(H["sim_ended"], ~H["gen_informed"])
            if np.any(returned_but_not_given):
                point_ids = np.where(returned_but_not_given)[0]
                Work[wid] = support.gen_work(
                    wid, gen_specs["persis_in"], point_ids, persis_info.get(wid), persistent=True
                )
                returned_but_not_given[point_ids] = False

    for wid in support.avail_worker_ids(persistent=False):
        # Skip any cancelled points
        while persis_info["next_to_give"] < len(H) and H[persis_info["next_to_give"]]["cancel_requested"]:
            persis_info["next_to_give"] += 1

        if persis_info["next_to_give"] < len(H):
            # perform sim evaluations (if they exist in History).
            try:
                Work[wid] = support.sim_work(wid, H, sim_specs["in"], persis_info["next_to_give"], persis_info.get(wid))
            except InsufficientFreeResources:
                break
            persis_info["next_to_give"] += 1

        elif persis_info.get("gen_started") is None:
            # Finally, call a persistent generator as there is nothing else to do.
            persis_info.get(wid)["nworkers"] = len(W)
            try:
                Work[wid] = support.gen_work(
                    wid, gen_specs.get("in", []), range(len(H)), persis_info.get(wid), persistent=True
                )
            except InsufficientFreeResources:
                break
            persis_info["gen_started"] = True  # Must set after - in case break on resources

    return Work, persis_info
