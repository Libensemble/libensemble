import copy
import logging
import numpy as np
from libensemble.resources.resources import Resources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class InsufficientResourcesError(Exception):
    "Raised when more resources are requested than exist"


class InsufficientFreeResources(Exception):
    "Raised when the requested resources could not be provisioned"


class ResourceScheduler:
    """Calculates and returns resource set ids from a dictionary of resource sets
    by group. Resource sets available is read from the resources module or from
    a resources object passed in."""

    def __init__(self, user_resources=None, sched_opts={}):
        """Initiate a ResourceScheduler object

        Parameters
        ----------

        user_resources: Resources object, optional
            A resources object. If present overrides the class variable.

        sched_opts: dict, optional
            A dictionary of scheduler options. Normally passed via alloc_specs['user']


        **Scheduler Options**:

        split2fit: boolean, optional
            Split across more nodes if space not currently avail (even though can fit when free).
            Default: True
        """

        self.resources = user_resources or Resources.resources.resource_manager
        self.rsets_free = self.resources.rsets_free
        self.avail_rsets_by_group = None

        # Process scheduler options
        self.split2fit = sched_opts.get('split2fit', True)

    def assign_resources(self, rsets_req):
        """Schedule resource sets to a work item if possible

        If the resources required are less than one node, they will be
        allocated to the smallest available sufficient slot.

        If the resources required are more than one node, then will
        attempt to find an even split. If no even split is possible,
        then enough full nodes will be allocated to cover the requirement.

        Returns a list of resource sets ids. A return of None implies
        insufficient resources.
        """

        if rsets_req > self.resources.total_num_rsets:
            raise InsufficientResourcesError("More resource sets requested {} than exist {}"
                                             .format(rsets_req, self.resources.total_num_rsets))

        if rsets_req > self.rsets_free:
            raise InsufficientFreeResources

        num_groups = self.resources.num_groups
        max_grpsize = self.resources.rsets_per_node  # assumes even
        avail_rsets_by_group = self.get_avail_rsets_by_group()

        # Work out best target fit - if all rsets were free.
        rsets_req, num_groups_req, rsets_req_per_group = \
            self.calc_req_split(rsets_req, max_grpsize, num_groups, extend=True)

        if self.split2fit:
            sorted_lengths = ResourceScheduler.get_sorted_lens(avail_rsets_by_group)
            max_even_grpsize = sorted_lengths[num_groups_req - 1]
            if max_even_grpsize == 0 and rsets_req > 0:
                raise InsufficientFreeResources
            if max_even_grpsize < rsets_req_per_group:
                # Cannot fit in smallest number of nodes - try to split
                rsets_req, num_groups_req, rsets_req_per_group = \
                    self.calc_even_split_uneven_groups(max_even_grpsize, num_groups_req,
                                                       rsets_req, sorted_lengths, num_groups, extend=False)
        tmp_avail_rsets_by_group = copy.deepcopy(avail_rsets_by_group)
        max_upper_bound = max_grpsize + 1

        # Now find slots on as many nodes as need
        accum_team = []
        group_list = []

        for ng in range(num_groups_req):
            cand_team, cand_group = \
                self.find_candidate(tmp_avail_rsets_by_group, group_list, rsets_req_per_group, max_upper_bound)

            if cand_group is not None:
                accum_team.extend(cand_team)
                group_list.append(cand_group)

                for rset in cand_team:
                    tmp_avail_rsets_by_group[cand_group].remove(rset)

        if len(accum_team) == rsets_req:
            # A successful team found
            rset_team = sorted(accum_team)
            self.avail_rsets_by_group = tmp_avail_rsets_by_group
            self.rsets_free -= rsets_req
        else:
            raise InsufficientFreeResources

        return rset_team

    def find_candidate(self, rsets_by_group, group_list, rsets_req_per_group, max_upper_bound):
        """Find a candidate slot in a group"""
        cand_team = []
        cand_group = None
        upper_bound = max_upper_bound
        for g in rsets_by_group:
            if g in group_list:
                continue
            nslots = len(rsets_by_group[g])
            if nslots == rsets_req_per_group:
                # Exact fit
                cand_team = rsets_by_group[g].copy()
                cand_group = g
                break
            elif rsets_req_per_group < nslots < upper_bound:
                cand_team = rsets_by_group[g][:rsets_req_per_group]
                cand_group = g
                upper_bound = nslots
        return cand_team, cand_group

    def get_avail_rsets_by_group(self):
        """Return a dictionary of resource set IDs for each group (e.g. node)

        If groups are not set they will all be in one group (group 0)

        E.g: Say 8 resource sets / 2 nodes
        GROUP  1: [1,2,3,4]
        GROUP  2: [5,6,7,8]
        """
        if self.avail_rsets_by_group is None:
            rsets = self.resources.rsets
            groups = np.unique(rsets['group'])
            self.avail_rsets_by_group = {}
            for g in groups:
                self.avail_rsets_by_group[g] = []
            for ind, rset in enumerate(rsets):
                if not rset['assigned']:
                    g = rset['group']
                    self.avail_rsets_by_group[g].append(ind)
        return self.avail_rsets_by_group

    def calc_req_split(self, rsets_req, max_grpsize, num_groups, extend):
        if self.resources.even_groups:  # This is total group sizes even (not available sizes)
            rsets_req, num_groups_req, rsets_req_per_group = \
                self.calc_rsets_even_grps(rsets_req, max_grpsize, num_groups, extend)
        else:
            logger.warning('Uneven groups - but using even groups function')
            rsets_req, num_groups_req, rsets_req_per_group = \
                self.calc_rsets_even_grps(rsets_req, max_grpsize, num_groups, extend)
        return rsets_req, num_groups_req, rsets_req_per_group

    def calc_rsets_even_grps(self, rsets_req, max_grpsize, max_groups, extend):
        """Calculate an even breakdown to best fit rsets_req input"""
        if rsets_req == 0:
            return 0, 0, 0

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
                if extend:
                    rsets_req_per_group = rsets_req//num_groups_req + (rsets_req % num_groups_req > 0)
                    rsets_req = num_groups_req * rsets_req_per_group
                    logger.debug('Increasing resource requirement to obtain an even partition of resource sets'
                                 '\nto nodes. rsets_req {}  num_groups_req {} rsets_req_per_group {}'.
                                 format(rsets_req, num_groups_req, rsets_req_per_group))
                else:
                    rsets_req_per_group = max_grpsize
        else:
            rsets_req_per_group = rsets_req
        return rsets_req, num_groups_req, rsets_req_per_group

    def calc_even_split_uneven_groups(self, rsets_per_grp, ngroups, rsets_req, sorted_lens, max_grps, extend):
        """Calculate an even breakdown to best fit rsets_req with uneven groups"""
        if rsets_req == 0:
            return 0, 0, 0

        while rsets_per_grp * ngroups != rsets_req:
            if rsets_per_grp * ngroups > rsets_req:
                rsets_per_grp -= 1
            else:
                ngroups += 1
                if ngroups > max_grps:
                    raise InsufficientFreeResources
                rsets_per_grp = sorted_lens[ngroups - 1]

        return rsets_req, ngroups, rsets_per_grp

    @staticmethod
    def get_sorted_lens(avail_rsets):
        """Get max length of a list value in a dictionary"""
        return sorted([len(v) for v in avail_rsets.values()], reverse=True)
