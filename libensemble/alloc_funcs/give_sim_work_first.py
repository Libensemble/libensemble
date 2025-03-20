import time

import numpy as np
import numpy.typing as npt

from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def give_sim_work_first(
    W: npt.NDArray,
    H: npt.NDArray,
    sim_specs: dict,
    gen_specs: dict,
    alloc_specs: dict,
    persis_info: dict,
    libE_info: dict,
) -> tuple[dict]:
    """
    Decide what should be given to workers. This allocation function gives any
    available simulation work first, and only when all simulations are
    completed or running does it start (at most ``alloc_specs["user"]["num_active_gens"]``)
    generator instances.

    Allows for a ``alloc_specs["user"]["batch_mode"]`` where no generation
    work is given out unless all entries in ``H`` are returned.

    Can give points in highest priority, if ``"priority"`` is a field in ``H``.
    If ``alloc_specs["user"]["give_all_with_same_priority"]`` is set to True, then
    all points with the same priority value are given as a batch to the sim.

    Workers performing sims will be assigned resources given in H["resource_sets"]
    this field exists, else defaulting to one. Workers performing gens are
    assigned resource_sets given by persis_info["gen_resources"] or zero.

    This is the default allocation function if one is not defined.

    tags: alloc, default, batch, priority

    .. seealso::
        `test_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_uniform_sampling.py>`_ # noqa
    """

    user = alloc_specs.get("user", {})

    if "cancel_sims_time" in user:
        # Cancel simulations that are taking too long
        rows = np.where(np.logical_and.reduce((H["sim_started"], ~H["sim_ended"], ~H["cancel_requested"])))[0]
        inds = time.time() - H["sim_started_time"][rows] > user["cancel_sims_time"]
        to_request_cancel = rows[inds]
        for row in to_request_cancel:
            H[row]["cancel_requested"] = True

    if libE_info["sim_max_given"] or not libE_info["any_idle_workers"]:
        return {}, persis_info

    # Initialize alloc_specs["user"] as user.
    batch_give = user.get("give_all_with_same_priority", False)
    gen_in = gen_specs.get("in", [])

    manage_resources = libE_info["use_resource_sets"]
    support = AllocSupport(W, manage_resources, persis_info, libE_info)
    gen_count = support.count_gens()
    Work = {}

    points_to_evaluate = ~H["sim_started"] & ~H["cancel_requested"]

    if np.any(points_to_evaluate):
        for wid in support.avail_worker_ids(gen_workers=False):
            sim_ids_to_send = support.points_by_priority(H, points_avail=points_to_evaluate, batch=batch_give)
            try:
                Work[wid] = support.sim_work(wid, H, sim_specs["in"], sim_ids_to_send, persis_info.get(wid))
            except InsufficientFreeResources:
                break
            points_to_evaluate[sim_ids_to_send] = False
            if not np.any(points_to_evaluate):
                break
    else:
        for wid in support.avail_worker_ids(gen_workers=True):
            # Allow at most num_active_gens active generator instances
            if gen_count >= user.get("num_active_gens", gen_count + 1):
                break

            # Do not start gen instances in batch mode if workers still working
            if user.get("batch_mode") and not support.all_sim_ended(H):
                break

            # Give gen work
            return_rows = range(len(H)) if gen_in else []
            try:
                Work[wid] = support.gen_work(wid, gen_in, return_rows, persis_info.get(wid))
            except InsufficientFreeResources:
                break
            gen_count += 1

    return Work, persis_info
