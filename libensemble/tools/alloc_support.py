import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.resources.resources import Resources

# SH TODO: May be move the more advanced functions below sim_work/gen_work?
#          Should add check that req. resource sets not larger than whole allocation.

class AllocException(Exception):
    "Raised for any exception in the alloc support"


def get_groupsize_from_resources():
    """Gets groups size from resources

    If resources is not set, returns None
    """
    resources = Resources.resources
    if resources is None:
        return None
    group_size = resources.managerworker_resources.workers_per_node
    return group_size


# SH TODO: An alt. could be to return an object
#          When merge CWP branch - will need option to use active receive worker
def get_avail_workers_by_group(W, persistent=None, zero_resource_workers=None):
    """Return a dictionary of workers IDs for each group (e.g. node)

    If groups are not set they will all be in one group (group 0)

    A zero_resource_worker is in group -1 (see resources.py).

    E.g: Say 9 workers / 2 nodes / 1 is zero_resource_worker (for gen).
    GROUP -1: [1]
    GROUP  1: [2,3,4,5]
    GROUP  2: [6,7,8,9]

    If zero_resource_workers=False, will return (cannot use worker 1):
    GROUP -1: []
    GROUP  1: [2,3,4,5]
    GROUP  2: [6,7,8,9]

    If zero_resource_workers=True, will return (can only use worker 1):
    GROUP -1: [1]
    GROUP  1: []
    GROUP  2: []

    Similar filtering is possible for workers in (or not in) a persistent state.
    """

    def fltr(wrk, field, option):
        """Filter by condition if supplied"""
        if option is None:
            return True
        return wrk[field] == option

    # For abbrev.
    def fltr_persis():
        if persistent is None:
            return True
        return wrk['persis_state'] == persistent

    def fltr_zrw():
        if zero_resource_workers is None:
            return True
        return wrk['zero_resource_worker'] == zero_resource_workers

    groups = np.unique(W['worker_group'])
    wrks_by_group = {}
    for g in groups:
        wrks_by_group[g] = []

    for wrk in W:
        if not wrk['blocked'] and not wrk['active'] and fltr_persis() and fltr_zrw():
            g = wrk['worker_group']
            wrks_by_group[g].append(wrk['worker_id'])

    return wrks_by_group


def update_avail_workers_by_group(wrks_by_group, worker_team, group_list=None):
    """Removes members of worker_team from wrks_by_group"""
    if group_list is None:
        group_list = wrks_by_group.keys()

    rm_check = []
    for g in group_list:
        for wrk in worker_team:
            if wrk in wrks_by_group[g]:
                wrks_by_group[g].remove(wrk)
                rm_check.append(wrk)

    assert set(worker_team) == set(rm_check), \
        "Error removing workers from available workers by group list"


# SH TODO: Naming - assign_workers?/schedule_workers?
#          There may be various scheduling options.
#          Terminology - E.g: What do we call the H row/s we are sending to the worker? work_item?
def assign_workers(wrks_by_group, nworkers_req):
    """Schedule worker resources to a work item.

    This routine assigns the resources of {nworkers_req} workers.

    If the resources required are less than one node, they will be
    allocated to the smallest available sufficient slot.

    If the resources required are more than one node, then enough
    full nodes will be allocated to cover the requirement.

    The assigned resources are removed from wrks_by_group list.

    Returns a list of worker ids. An empty list
    implies insufficient resources.

    """
    gsize = get_groupsize_from_resources()
    if gsize is not None:
        upper_bound = gsize + 1
    else:
        # All in group zero
        # Will still block workers, but treats all as one group
        if len(wrks_by_group) > 1:
            raise AllocException("There should only be one group if resources is not set")
        upper_bound = len(wrks_by_group[0]) + 1

    # SH TODO: Review scheduling > 1 node strategy
    # If using more than one node - round up to full nodes
    # alternative, could be to allow the allocation function to do this.
    # also alt. could be to round up to next even split (e.g. 5 workers beccomes 6 - and get 2 groups of 3).
    num_groups = nworkers_req//gsize + (nworkers_req % gsize > 0)  # Divide with roundup.
    nworkers_per_group = nworkers_req
    if num_groups > 1:
        nworkers_req = num_groups * gsize
        nworkers_per_group = gsize

    # Now find slots on as many nodes as need
    worker_team = []
    accum_team = []
    group_list = []
    for ng in range(num_groups):
        cand_team = []
        cand_group = None
        for g in wrks_by_group:
            if g in group_list:
                continue
            nslots = len(wrks_by_group[g])
            if nslots == nworkers_per_group:  # Exact fit.
                cand_team = wrks_by_group[g].copy()  # SH TODO: check do I still need copy - given extend below?
                cand_group = g
                break  # break out inner loop...
            elif nworkers_per_group < nslots < upper_bound:
                cand_team = wrks_by_group[g][:nworkers_per_group]
                cand_group = g
                upper_bound = nslots
        if cand_group is not None:
            accum_team.extend(cand_team)
            group_list.append(cand_group)

    if len(accum_team) == nworkers_req:
        # A successful team found
        worker_team = accum_team

    # SH TODO: Remove when done testing
    #if len(worker_team) > 4:
        #print('Worker team', worker_team)

    update_avail_workers_by_group(wrks_by_group, worker_team, group_list=group_list)
    return worker_team


def get_worker_group(W, workerID):
    return W['worker_group'][W['worker_id'] == workerID]  # Should be unique


def avail_worker_ids(W, persistent=None):
    """Returns available workers (``active == 0``), as an array, filtered by ``persis_state``.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    :param persistent: Optional Boolean. If specified, also return workers with given persis_state.
    """
    # SH TODO: update docstring

    if persistent is None:
        return W['worker_id'][W['active'] == 0]
    if persistent:
        return W['worker_id'][np.logical_and(W['active'] == 0,
                                             W['persis_state'] != 0)]

    return W['worker_id'][np.logical_and(W['active'] == 0,
                                         W['persis_state'] == 0)]


def count_gens(W):
    """Return the number of active generators in a set of workers.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    """
    return sum(W['active'] == EVAL_GEN_TAG)


def test_any_gen(W):
    """Return True if a generator worker is active.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    """
    return any(W['active'] == EVAL_GEN_TAG)


def count_persis_gens(W):
    """Return the number of active persistent generators in a set of workers.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    """
    return sum(W['persis_state'] == EVAL_GEN_TAG)


def sim_work(Work, i, H_fields, H_rows, persis_info, **libE_info):
    """Add sim work record to given Work array.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    :param i: Worker ID.
    :param H_fields: Which fields from H to send
    :param persis_info: current persis_info dictionary

    :returns: None
    """
    libE_info['H_rows'] = H_rows
    Work[i] = {'H_fields': H_fields,
               'persis_info': persis_info,
               'tag': EVAL_SIM_TAG,
               'libE_info': libE_info}


# SH TODO: Variant accepting worker_team - need to test
#          This may replace sim work as is does the blocking
def sim_work_with_blocking(Work, worker_team, H_fields, H_rows, persis_info, **libE_info):
    """Add sim work record to given Work array.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    :param i: Worker ID.
    :param H_fields: Which fields from H to send
    :param persis_info: current persis_info dictionary

    :returns: None
    """
    if isinstance(worker_team, list):
        worker = worker_team[0]
        if len(worker_team) > 1:
            libE_info['blocking'] = worker_team[1:]
    else:
        worker = worker_team

    libE_info['H_rows'] = H_rows
    Work[worker] = {'H_fields': H_fields,
                    'persis_info': persis_info,
                    'tag': EVAL_SIM_TAG,
                    'libE_info': libE_info}


def gen_work(Work, i, H_fields, H_rows, persis_info, **libE_info):
    """Add gen work record to given Work array.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    :param i: Worker ID.
    :param H_fields: Which fields from H to send
    :param persis_info: current persis_info dictionary

    :returns: None
    """
    libE_info['H_rows'] = H_rows
    Work[i] = {'H_fields': H_fields,
               'persis_info': persis_info,
               'tag': EVAL_GEN_TAG,
               'libE_info': libE_info}
