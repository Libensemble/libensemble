from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def give_pregenerated_sim_work(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):
    """
    This allocation function gives (in order) entries in alloc_spec['x'] to
    idle workers. It is an example use case where no gen_func is used.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
        return {}, persis_info

    manage_resources = "resource_sets" in H.dtype.names or libE_info["use_resource_sets"]
    support = AllocSupport(W, manage_resources, persis_info, libE_info)
    Work = {}

    # Unless already defined, initialize next_to_give to be the first point in H
    persis_info["next_to_give"] = persis_info.get("next_to_give", 0)

    if persis_info["next_to_give"] >= len(H):
        return Work, persis_info, 1

    for i in support.avail_worker_ids():
        # Skip any cancelled points
        while persis_info["next_to_give"] < len(H) and H[persis_info["next_to_give"]]["cancel_requested"]:
            persis_info["next_to_give"] += 1

        # Give sim work
        try:
            Work[i] = support.sim_work(i, H, sim_specs["in"], [persis_info["next_to_give"]], [])
        except InsufficientFreeResources:
            break
        persis_info["next_to_give"] += 1

        if persis_info["next_to_give"] >= len(H):
            break

    return Work, persis_info
