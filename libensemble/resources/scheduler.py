import copy
import itertools
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
    by group. The available resource sets are read initially from the resources
    module or from a resources object passed in.

    Resource sets are locally provisioned to work items by a call to the
    ``assign_resources`` function, and a cache of available resource sets is
    maintained for the life of the object (usually corresponding to one call
    of the allocation function). Note that work item resources are formally
    assigned to workers only when a work item is sent to the worker.
    """

    def __init__(self, user_resources=None, sched_opts={}):
        """Initiate a ResourceScheduler object

        Parameters
        ----------

        user_resources: Resources, optional
            A resources object. If present overrides the class variable.

        sched_opts: dict, optional
            A dictionary of scheduler options. Passed via ``libE_specs["scheduler_opts"]``


        The supported fields for sched_opts are::

            "split2fit" [Boolean]:
                Try to split resource sets across more nodes if space is not currently
                available on the minimum node count required. Allows more efficient
                scheduling.
                Default: True

            "match_slots" [Boolean]:
                When splitting resource sets across multiple nodes, slot IDs must match.
                Useful if setting an environment variable such as ``CUDA_VISIBLE_DEVICES``
                to specific slots, which should match over multiple nodes.
                Default: True
        """

        self.resources = user_resources or Resources.resources.resource_manager

        self.rsets_free = self.resources.rsets_free
        self.gpu_rsets_free = self.resources.gpu_rsets_free
        self.nongpu_rsets_free = self.resources.nongpu_rsets_free

        self.avail_rsets_by_group = None
        self.log_msg = None

        # Process scheduler options
        self.split2fit = sched_opts.get("split2fit", True)
        self.match_slots = sched_opts.get("match_slots", True)
        self.last_use_gpus = None

    def assign_resources(self, rsets_req, use_gpus=None, user_params=[]):
        """Schedule resource sets to a work item if possible.

        If the resources required are less than one node, they will be
        allocated to the smallest available sufficient slot.

        If the resources required are more than one node, then
        the scheduler will attempt to find an even split. If no even split
        is possible, then enough additional resource sets will be
        assigned to enable an even split.

        Returns a list of resource set IDs or raises an exception (either
        InsufficientResourcesError or InsufficientFreeResources).
        """

        if rsets_req == 0:
            return []

        self.check_total_rsets(rsets_req, use_gpus)
        self.log_msg = None  # Log resource messages only when find resources
        num_groups = self.resources.num_groups

        max_grpsize = self.resources.rsets_per_node  # assumes even
        if use_gpus is not None:
            if use_gpus:
                max_grpsize = self.resources.gpu_rsets_per_node
            else:
                max_grpsize = self.resources.nongpu_rsets_per_node

        avail_rsets_by_group = self.get_avail_rsets_by_group()

        # get rsets by group that are of the required type
        valid_rsets_by_group = self.filter_for_rset_type(avail_rsets_by_group, use_gpus)

        try_split = self.split2fit

        # Work out best target fit - if all rsets were free.
        rsets_req, num_groups_req, rsets_req_per_group = self.calc_req_split(
            rsets_req, max_grpsize, num_groups, extend=True
        )

        # Check enough slots
        sorted_lengths = ResourceScheduler.get_sorted_lens(valid_rsets_by_group)
        max_even_grpsize = sorted_lengths[num_groups_req - 1]
        if max_even_grpsize < rsets_req_per_group:
            if not self.split2fit or max_even_grpsize == 0:
                raise InsufficientFreeResources

        if self.match_slots:
            slots_avail_by_group = self.get_avail_slots_by_group(valid_rsets_by_group)
            cand_groups, cand_slots = self.get_matching_slots(slots_avail_by_group, num_groups_req, rsets_req_per_group)

            if cand_groups is None:
                if not self.split2fit:
                    raise InsufficientFreeResources
            else:
                try_split = False  # Already found

        if try_split:
            if max_even_grpsize < rsets_req_per_group:
                found_split = False
                while not found_split:
                    # Finds a split with enough slots (not nec. matching slots) if exists.
                    rsets_req, num_groups_req, rsets_req_per_group = self.calc_even_split_uneven_groups(
                        max_even_grpsize, num_groups_req, rsets_req, sorted_lengths, num_groups, user_params
                    )
                    if self.match_slots:
                        cand_groups, cand_slots = self.get_matching_slots(
                            slots_avail_by_group, num_groups_req, rsets_req_per_group
                        )
                        if cand_groups is not None:
                            found_split = True
                        else:
                            num_groups_req += 1  # try one more group
                    else:
                        found_split = True

        if self.match_slots:
            if cand_groups is None:
                raise InsufficientFreeResources
            else:
                rset_team = self.assign_team_from_slots(
                    slots_avail_by_group, cand_groups, cand_slots, rsets_req_per_group
                )
        else:
            rset_team = self.find_rsets_any_slots(
                valid_rsets_by_group, max_grpsize, rsets_req, num_groups_req, rsets_req_per_group
            )

        # Update persistent attributes
        self.avail_rsets_by_group = self.filter_out_rset_team(avail_rsets_by_group, rset_team)
        self.rsets_free -= len(rset_team)
        if use_gpus is not None:
            if use_gpus:
                self.gpu_rsets_free -= len(rset_team)
            else:
                self.nongpu_rsets_free -= len(rset_team)

        if self.log_msg is not None:
            logger.debug(self.log_msg)

        logger.debug(
            f"rset_team found: Req: {rsets_req} rsets. Found: {rset_team} Avail sets {self.avail_rsets_by_group}"
        )

        return rset_team

    def find_rsets_any_slots(self, valid_rsets_by_group, max_grpsize, rsets_req, ngroups, rsets_per_group):
        """Find optimal non-matching slots across groups"""
        tmp_rsets_by_group = copy.deepcopy(valid_rsets_by_group)
        max_upper_bound = max_grpsize + 1

        # Now find slots on as many nodes as need
        accum_team = []
        group_list = []

        for ng in range(ngroups):
            cand_team, cand_group = self.find_candidate(
                tmp_rsets_by_group, group_list, rsets_per_group, max_upper_bound
            )

            if cand_group is not None:
                accum_team.extend(cand_team)
                group_list.append(cand_group)

                for rset in cand_team:
                    tmp_rsets_by_group[cand_group].remove(rset)

        if len(accum_team) == rsets_req:
            # A successful team found
            rset_team = sorted(accum_team)
        else:
            raise InsufficientFreeResources
        return rset_team

    def find_candidate(self, rsets_by_group, group_list, rsets_per_group, max_upper_bound):
        """Find a candidate slot in a group"""
        cand_team = []
        cand_group = None
        upper_bound = max_upper_bound
        for g in rsets_by_group:
            if g in group_list:
                continue
            nslots = len(rsets_by_group[g])
            if nslots == rsets_per_group:
                # Exact fit
                cand_team = rsets_by_group[g].copy()
                cand_group = g
                break
            elif rsets_per_group < nslots < upper_bound:
                cand_team = rsets_by_group[g][:rsets_per_group]
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
            groups = np.unique(rsets["group"])
            self.avail_rsets_by_group = {}
            for g in groups:
                self.avail_rsets_by_group[g] = []
            for ind, rset in enumerate(rsets):
                if rset["assigned"] == -1:  # now default is -1.
                    g = rset["group"]
                    self.avail_rsets_by_group[g].append(ind)
        return self.avail_rsets_by_group

    def filter_for_rset_type(self, avail_rsets_by_group, use_gpus):
        """Return avail_rsets_by_group filtered by rset type (gpus/non-gpus/all)"""

        def fltr_gpus(rset):
            """Filter rsets to remove - hence negation"""
            if use_gpus:
                return ~rset["gpus"]
            else:
                return rset["gpus"]

        if use_gpus is None:
            return avail_rsets_by_group  # this means will be same object - dont need to update at end!

        valid_rsets_by_group = {}
        for g, rlist in avail_rsets_by_group.items():
            valid_rsets = rlist.copy()
            for rid in rlist:
                if fltr_gpus(self.resources.rsets[rid]):
                    valid_rsets.remove(rid)
            valid_rsets_by_group[g] = valid_rsets

        return valid_rsets_by_group

    def filter_out_rset_team(self, avail_rsets_by_group, rset_team):
        """Return avail_rsets_by_group filtered by rset type (gpus/non-gpus/all)"""
        for g, rlist in avail_rsets_by_group.items():
            valid_rsets = rlist.copy()
            for rid in rlist:
                if rid in rset_team:
                    valid_rsets.remove(rid)
            avail_rsets_by_group[g] = valid_rsets
        return avail_rsets_by_group

    @staticmethod
    def get_slots_of_len(d, n):
        """Filter dictionary to values >= n"""
        return {k: v for k, v in d.items() if len(v) >= n}

    def get_avail_slots_by_group(self, rsets_by_group):
        """Return a dictionary of free slot IDS for each group (e.g. node)"""
        slots_avail_by_group = {}
        for k, v in rsets_by_group.items():
            slots_avail_by_group[k] = set([self.resources.rsets[i]["slot"] for i in v])
        return slots_avail_by_group

    def calc_req_split(self, rsets_req, max_grpsize, num_groups, extend):
        rsets_req, num_groups_req, rsets_per_group = self.calc_rsets_even_grps(
            rsets_req, max_grpsize, num_groups, extend
        )
        return rsets_req, num_groups_req, rsets_per_group

    def calc_rsets_even_grps(self, rsets_req, max_grpsize, max_groups, extend):
        """Calculate an even breakdown to best fit rsets_req input"""
        if rsets_req == 0:
            return 0, 0, 0

        # Divide with roundup
        num_groups_req = rsets_req // max_grpsize + (rsets_req % max_grpsize > 0)

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
                rsets_per_group = rsets_req // num_groups_req  # This should always divide perfectly.
            else:
                if extend:
                    rsets_per_group = rsets_req // num_groups_req + (rsets_req % num_groups_req > 0)
                    orig_rsets_req = rsets_req
                    rsets_req = num_groups_req * rsets_per_group
                    self.log_msg = (
                        "Increasing resource requirement to obtain an even partition of resource sets\n"
                        f"to nodes. rsets_req orig: {orig_rsets_req} New: {rsets_req} "
                        f"  num_groups_req {num_groups_req} rsets_per_group {rsets_per_group}"
                    )
                else:
                    rsets_per_group = max_grpsize
        else:
            rsets_per_group = rsets_req
        return rsets_req, num_groups_req, rsets_per_group

    def check_params(self, user_params, ngroups):
        """Return True if all user params divide by number of groups, else False"""
        for param in user_params:
            if param % ngroups != 0:
                return False
        return True

    def calc_even_split_uneven_groups(self, rsets_per_grp, ngroups, rsets_req, sorted_lens, max_grps, user_params):
        """Calculate an even breakdown to best fit rsets_req with uneven groups"""
        while rsets_per_grp * ngroups != rsets_req or not self.check_params(user_params, ngroups):
            if rsets_per_grp * ngroups > rsets_req:
                rsets_per_grp -= 1
            else:
                ngroups += 1
                if ngroups > max_grps:
                    raise InsufficientFreeResources
                rsets_per_grp = sorted_lens[ngroups - 1]
        return rsets_req, ngroups, rsets_per_grp

    def assign_team_from_slots(self, slots_avail_by_group, cand_groups, cand_slots, rsets_per_group):
        """Assign resource set team from slots"""
        rset_team = []
        for grp in cand_groups:
            for i, slot in enumerate(cand_slots):
                # Ignore extra slots
                if i >= rsets_per_group:
                    break
                group = self.resources.rsets["group"] == grp
                slot = self.resources.rsets["slot"] == slot
                rset = np.nonzero(group & slot)[0][0]
                rset_team.append(rset)
        return sorted(rset_team)

    @staticmethod
    def get_sorted_lens(avail_rsets):
        """Get max length of a list value in a dictionary"""
        return sorted([len(v) for v in avail_rsets.values()], reverse=True)

    def get_matching_slots(self, slots_avail_by_group, num_groups_req, rsets_per_group):
        """Get first N matching slots across groups

        Assumes num_groups_req > 0.
        """
        viable_slots_by_group = ResourceScheduler.get_slots_of_len(slots_avail_by_group, rsets_per_group)
        if len(viable_slots_by_group) < num_groups_req:
            return None, None

        combs = itertools.combinations(viable_slots_by_group, num_groups_req)
        cand_groups = None
        cand_slots = None
        upper_bound = max(len(v) for v in viable_slots_by_group.values()) + 1
        for comb in combs:
            tmplist = []
            for i in comb:
                tmplist.append(viable_slots_by_group[i])
            common_slots = set.intersection(*tmplist)
            if len(common_slots) == rsets_per_group:
                cand_groups = comb
                cand_slots = common_slots
                break
            elif rsets_per_group < len(common_slots) < upper_bound:
                upper_bound = len(common_slots)
                cand_groups = comb
                cand_slots = common_slots

        if cand_groups is None:
            return None, None

        return cand_groups, cand_slots

    def check_total_rsets(self, rsets_req, use_gpus):
        """Raise exceptions if rsets requested is more than total that exist or available"""
        if use_gpus is None:
            nrsets = self.resources.total_num_rsets
            if rsets_req > nrsets:
                raise InsufficientResourcesError(f"More resource sets requested {rsets_req} than exist {nrsets}")
        elif use_gpus:
            nrsets = self.resources.total_num_gpu_rsets
            if rsets_req > nrsets:
                raise InsufficientResourcesError(f"More GPU resource sets requested {rsets_req} than exist {nrsets}")
        else:
            nrsets = self.resources.total_num_nongpu_rsets
            if rsets_req > nrsets:
                raise InsufficientResourcesError(
                    f"More non-GPU resource sets requested {rsets_req} than exist {nrsets}"
                )

        if use_gpus is None:
            if rsets_req > self.rsets_free:
                raise InsufficientFreeResources
        elif use_gpus:
            if rsets_req > self.gpu_rsets_free:
                raise InsufficientFreeResources
        else:
            if rsets_req > self.nongpu_rsets_free:
                raise InsufficientFreeResources
