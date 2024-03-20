from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def ensure_one_active_gen(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs["in"]``
    are given. If there is no active generator, then one is started.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_fast_alloc.py>`_ # noqa
    """

    if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
        return {}, persis_info

    manage_resources = libE_info["use_resource_sets"]
    support = AllocSupport(W, manage_resources, persis_info, libE_info)

    Work = {}
    gen_flag = True
    gen_in = gen_specs.get("in", [])

    if persis_info["next_to_give"] < len(H):
        for wid in support.avail_worker_ids(gen_workers=False):
            persis_info = support.skip_canceled_points(H, persis_info)
            try:
                Work[wid] = support.sim_work(wid, H, sim_specs["in"], [persis_info["next_to_give"]], [])
            except InsufficientFreeResources:
                break
            persis_info["next_to_give"] += 1
            if persis_info["next_to_give"] >= len(H):
                break

    elif not support.test_any_gen() and gen_flag:
        # Give gen work
        return_rows = range(len(H)) if gen_in else []
        wid = support.avail_worker_ids(gen_workers=True)[0]

        if not support.all_sim_ended(H):
            return Work, persis_info

        try:
            Work[wid] = support.gen_work(wid, gen_in, return_rows, persis_info.get(wid))
        except InsufficientFreeResources:
            return Work, persis_info
        gen_flag = False
        persis_info["total_gen_calls"] += 1

    return Work, persis_info
