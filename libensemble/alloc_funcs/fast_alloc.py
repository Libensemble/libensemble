from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info, libE_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If all entries in `H` have been given a be evaluated, a worker
    is told to call the generator function, provided this wouldn't result in
    more than ``alloc_specs['user']['num_active_gen']`` active generators.

    This fast_alloc variation of give_sim_work_first is useful for cases that
    simply iterate through H, issuing evaluations in order and, in particular,
    is likely to be faster if there will be many short simulation evaluations,
    given that this function contains fewer column length operations.

    tags: alloc, simple, fast

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
        return {}, persis_info

    user = alloc_specs.get("user", {})
    manage_resources = "resource_sets" in H.dtype.names or libE_info["use_resource_sets"]

    support = AllocSupport(W, manage_resources, persis_info, libE_info)

    gen_count = support.count_gens()
    Work = {}
    gen_in = gen_specs.get("in", [])

    for wid in support.avail_worker_ids():
        # Skip any cancelled points
        while persis_info["next_to_give"] < len(H) and H[persis_info["next_to_give"]]["cancel_requested"]:
            persis_info["next_to_give"] += 1

        # Give sim work if possible
        if persis_info["next_to_give"] < len(H):
            try:
                Work[wid] = support.sim_work(wid, H, sim_specs["in"], [persis_info["next_to_give"]], [])
            except InsufficientFreeResources:
                break
            persis_info["next_to_give"] += 1

        elif gen_count < user.get("num_active_gens", gen_count + 1):

            # Give gen work
            return_rows = range(len(H)) if gen_in else []
            try:
                Work[wid] = support.gen_work(wid, gen_in, return_rows, persis_info.get(wid))
            except InsufficientFreeResources:
                break
            gen_count += 1
            persis_info["total_gen_calls"] += 1

    return Work, persis_info
