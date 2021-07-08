import copy
import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.resources.resources import Resources

# SH TODO: May be move the more advanced functions below sim_work/gen_work?
#          Should add check that req. resource sets not larger than whole allocation.


class AllocException(Exception):
    "Raised for any exception in the alloc support"


# SH TODO: Not using now - but need this to work if resources is None
#def get_groupsize_from_resources(resources):
    #"""Gets groups size from resources

    #If resources is not set, returns None
    #"""
    ##resources = Resources.resources
    ##if resources is None:
        ##return None
    #group_size = resources.rsets_per_node
    ## print('groupsize is', group_size, flush=True)  # SH TODO:Remove
    #return group_size


# SH TODO: An alt. could be to return an object
#          When merge CWP branch - will need option to use active receive worker
def get_avail_rsets_by_group(resources):
    """Return a dictionary of resource set IDs for each group (e.g. node)

    If groups are not set they will all be in one group (group 0)

    E.g: Say 8 resource sets / 2 nodes
    GROUP  1: [1,2,3,4]
    GROUP  2: [5,6,7,8]
    """

    #resources = Resources.resources.managerworker_resources
    rsets = resources.rsets

    # SH TODO: Constructing rsets_by_group every call - look at storing in this format.
    groups = np.unique(rsets['group'])
    rsets_by_group = {}
    for g in groups:
        rsets_by_group[g] = []

    for ind, rset in enumerate(rsets):
        if not rset['assigned']:
            g = rset['group']
            rsets_by_group[g].append(ind)
    return rsets_by_group


# SH TODO UPDATE: This is now defunct with resource sets but need to
#                 update all alloc funcs before remove
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


# SH TODO UPDATE: This is now defunct with resource sets but need to
#                 update all alloc funcs before remove
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


# This one assumes equal groups sizes
def calc_rsets_even_grps(rsets_req, max_grpsize, max_groups):
    """Calculate an even breakdown to best fit rsets_req input"""

    num_groups_req = rsets_req//max_grpsize + (rsets_req % max_grpsize > 0)  # Divide with roundup.

    # Up to max groups - keep trying for an even split
    if num_groups_req > 1:
        even_partition = False
        tmp_num_groups = num_groups_req
        while tmp_num_groups <= max_groups:
            if rsets_req % tmp_num_groups == 0:
                even_partition = True
                break
            tmp_num_groups += 1

        if even_partition:
            num_groups_req = tmp_num_groups
            rsets_req_per_group = rsets_req//num_groups_req  # This should always divide perfectly.
        else:
            #log here
            print('Warning: Increasing resource requirement to obtain an even partition of resource sets to nodes')
            rsets_req_per_group = rsets_req//num_groups_req + (rsets_req % num_groups_req > 0)
            rsets_req = num_groups_req * rsets_req_per_group
    else:
        rsets_req_per_group = rsets_req

    return rsets_req, num_groups_req, rsets_req_per_group



# SH TODO: Naming - assign_resources?/assign_rsets?
#          There may be various scheduling options.
#          Terminology - E.g: What do we call the H row/s we are sending to the worker? work_item?
def assign_resources(rsets_req, worker_id, user_resources=None):
    """Schedule resource sets to a work item if possible and assign to worker

    This routine assigns the resources given by {rsets_req} and gives to
    worker {worker_id}.

    If the resources required are less than one node, they will be
    allocated to the smallest available sufficient slot.

    If the resources required are more than one node, then enough
    full nodes will be allocated to cover the requirement.

    The assigned resources are marked directly in the resources module.

    Returns a list of resource sets ids. A return of None implies
    insufficient resources.
    """

    resources = user_resources or Resources.resources.managerworker_resources

    if rsets_req > resources.num_rsets:
        # Raise error - when added errors
        raise AllocException("More resource sets requested {} than exist {}".format(rsets_req, resources.num_rsets))
    num_groups = resources.num_groups
    max_grpsize = resources.rsets_per_node  #assumes even

    #when use class will do this once for an alloc call!
    avail_rsets_by_group = get_avail_rsets_by_group(resources)

    #Maybe more efficient way than making copy_back
    tmp_avail_rsets_by_group = copy.deepcopy(avail_rsets_by_group)

    # print('Available rsets_by_group:', avail_rsets_by_group)  # SH TODO: Remove

    #print('max_grpsize is', max_grpsize)
    if max_grpsize is not None:
        max_upper_bound = max_grpsize + 1
    else:
        # All in group zero
        # Will still block workers, but treats all as one group
        if len(tmp_avail_rsets_by_group) > 1:
            raise AllocException("There should only be one group if resources is not set")
        max_upper_bound = len(tmp_avail_rsets_by_group[0]) + 1

    # SH TODO: Review scheduling > 1 node strategy
    # Currently tries for even split and if cannot, then rounds rset up to full nodes.
    # Log if change requested to make fit/round up - at least at debug level.
    if resources.even_groups:
        rsets_req, num_groups_req, rsets_req_per_group = calc_rsets_even_grps(rsets_req, max_grpsize, num_groups)
    else:
        print('Warning: uneven groups - but using even groups function')
        rsets_req, num_groups_req, rsets_req_per_group = calc_rsets_even_grps(rsets_req, max_grpsize, num_groups)

    # Now find slots on as many nodes as need
    accum_team = []
    group_list = []
    #print('\nLooking for {} rsets'.format(rsets_req))
    for ng in range(num_groups_req):
        #print(' - Looking for group {} out of {}: Groupsize {}'.format(ng+1, num_groups_req, rsets_req_per_group))
        cand_team = []
        cand_group = None
        upper_bound = max_upper_bound
        for g in tmp_avail_rsets_by_group:
            #print('   -- Search possible group {} in {}'.format(g, tmp_avail_rsets_by_group))

            if g in group_list:
                continue
            nslots = len(tmp_avail_rsets_by_group[g])
            if nslots == rsets_req_per_group:  # Exact fit.  # If make array - could work with different sized group requirements.
                cand_team = tmp_avail_rsets_by_group[g].copy()  # SH TODO: check do I still need copy - given extend below?
                cand_group = g
                break  # break out inner loop...
            elif rsets_req_per_group < nslots < upper_bound:
                cand_team = tmp_avail_rsets_by_group[g][:rsets_req_per_group]
                cand_group = g
                upper_bound = nslots
        if cand_group is not None:
            accum_team.extend(cand_team)
            group_list.append(cand_group)

            #print('      here b4:  group {} avail {} - cand_team {}'.format(group_list, tmp_avail_rsets_by_group, cand_team))
            #import pdb;pdb.set_trace()
            for rset in cand_team:
                tmp_avail_rsets_by_group[cand_group].remove(rset)
            #print('      here aft: group {} avail {}'.format(group_list, tmp_avail_rsets_by_group))

    #print('Found rset team {} - group_list {}'.format(accum_team, group_list))
    #import pdb;pdb.set_trace()

    if len(accum_team) == rsets_req:
        # A successful team found
        rset_team = sorted(accum_team)
        #print('Setting rset team {} - group_list {}'.format(accum_team, group_list))
        avail_rsets_by_group = tmp_avail_rsets_by_group
    else:
        rset_team = None  # Insufficient resources to honor

    # print('Assigned rset team {} to worker {}'.format(rset_team,worker_id))  # SH TODO: Remove

    #SH TODO: As move to class this will be packed and a temporary buffer used in the alloc so that
    #work units can be cancelled - ie manager always assigns actual resources when sending out.
    resources.assign_rsets(rset_team, worker_id)

    return rset_team


# SH TODO UPDATE: This is now defunct with resource sets but need to
#                 update all alloc funcs before remove
def assign_workers(rsets_by_group, rsets_req):
    """Schedule worker resources to a work item.

    This routine assigns the resources of {rsets_req} workers.

    If the resources required are less than one node, they will be
    allocated to the smallest available sufficient slot.

    If the resources required are more than one node, then enough
    full nodes will be allocated to cover the requirement.

    The assigned resources are removed from rsets_by_group list.

    Returns a list of worker ids. An empty list
    implies insufficient resources.

    """

    gsize = get_groupsize_from_resources()

    if gsize is not None:
        upper_bound = gsize + 1
    else:
        # All in group zero
        # Will still block workers, but treats all as one group
        if len(rsets_by_group) > 1:
            raise AllocException("There should only be one group if resources is not set")
        upper_bound = len(rsets_by_group[0]) + 1

    # SH TODO: Review scheduling > 1 node strategy
    # If using more than one node - round up to full nodes
    # alternative, could be to allow the allocation function to do this.
    # also alt. could be to round up to next even split (e.g. 5 workers beccomes 6 - and get 2 groups of 3).
    num_groups = rsets_req//gsize + (rsets_req % gsize > 0)  # Divide with roundup.
    rsets_per_group = rsets_req
    if num_groups > 1:
        rsets_req = num_groups * gsize
        rsets_per_group = gsize

    # Now find slots on as many nodes as need
    accum_team = []
    group_list = []
    for ng in range(num_groups):
        cand_team = []
        cand_group = None
        for g in rsets_by_group:
            if g in group_list:
                continue
            nslots = len(rsets_by_group[g])
            if nslots == rsets_per_group:  # Exact fit.
                cand_team = rsets_by_group[g].copy()  # SH TODO: check do I still need copy - given extend below?
                cand_group = g
                break  # break out inner loop...
            elif rsets_per_group < nslots < upper_bound:
                cand_team = rsets_by_group[g][:rsets_per_group]
                cand_group = g
                upper_bound = nslots
        if cand_group is not None:
            accum_team.extend(cand_team)
            group_list.append(cand_group)

    if len(accum_team) == rsets_req:
        # A successful team found
        rset_team = accum_team

    update_avail_workers_by_group(rsets_by_group, rset_team, group_list=group_list)
    return rset_team


def avail_worker_ids(W, persistent=None, active_recv=False, zero_resource_workers=None):
    """Returns available workers as a list, filtered by the given options`.

    :param W: :doc:`Worker array<../data_structures/worker_array>`
    :param persistent: Optional int. If specified, only return workers with given persis_state.
    :param active_recv: Optional Boolean. Only return workers with given active_recv. Default False.
    :param zero_resource_workers: Optional Boolean. If specified, only return workers with given zrw value.

    If there are no zero resource workers defined, then the zero_resource_workers argument will
    be ignored.
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
        # If none exist or you did not ask for zrw then return True
        if no_zrw or zero_resource_workers is None:
            return True
        return wrk['zero_resource_worker'] == zero_resource_workers

    def fltr_recving():
        if active_recv:
            return wrk['active_recv']  # SH TODO: must be persistent - could check here
        else:
            return not wrk['active']

    if active_recv and not persistent:
        raise AllocException("Cannot ask for non-persistent active receive workers")

    # SH if there are no zero resource workers - then ignore zrw (i.e. use only if they exist)
    no_zrw = not any(W['zero_resource_worker'])
    wrks = []
    for wrk in W:
        # SH TODO 'blocked' condition to be removed.
        if not wrk['blocked'] and fltr_recving() and fltr_persis() and fltr_zrw():
            wrks.append(wrk['worker_id'])
    return wrks


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
    libE_info['H_rows'] = np.atleast_1d(H_rows)
    Work[i] = {'H_fields': H_fields,
               'persis_info': persis_info,
               'tag': EVAL_SIM_TAG,
               'libE_info': libE_info}


# SH TODO: Need to update for resource sets
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

    # Count total gens
    try:
        gen_work.gen_counter += 1
    except AttributeError:
        gen_work.gen_counter = 1
    libE_info['gen_count'] = gen_work.gen_counter

    libE_info['H_rows'] = np.atleast_1d(H_rows)
    Work[i] = {'H_fields': H_fields,
               'persis_info': persis_info,
               'tag': EVAL_GEN_TAG,
               'libE_info': libE_info}


def all_returned(H, pt_filter=True):
    """Check if all expected points have returned from sim

    :param H: A :doc:`history array<../data_structures/history_array>`
    :param pt_filter: Optional boolean array filtering expected returned points: Default: All True

    :returns: Boolean. True if all expected points have been returned
    """
    # Exclude cancelled points that were not already given out
    excluded_points = H['cancel_requested'] & ~H['given']
    return np.all(H['returned'][pt_filter & ~excluded_points])
