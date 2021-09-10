import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.resources.resources import Resources
from libensemble.resources.scheduler import ResourceScheduler, InsufficientFreeResources

# General aim is to not allow user options (via {sim/gen/alloc}_specs) to be hidden in here.
# One exception is scheduler_opts... Now I'm extracting outside and passing in.
# persis_info['gen_resources'] is here. Could move outside, but then have to pass requested rsets through to gen_work.....
# H and W are passed in for convenience on init. They should be used read-only.

class AllocException(Exception):
    "Raised for any exception in the alloc support"


class AllocSupport:
    """A helper class to assist with writing allocation functions, containing methods for
    common operations like populating work units, determining which workers are available,
    evaluating what values need to be distributed to workers, and others.
    Note that since the ``alloc_f`` is called periodically by the Manager, this
    class instance (if used) will be recreated/destroyed on each loop."""

    gen_counter = 0

    def __init__(self, W, H, persis_info={}, scheduler_opts={}, user_resources=None, user_scheduler=None):
        """Instantiate a new AllocSupport instance

        ``W`` and ``H`` are passed in on initiation. They are referenced by the various methods,
        but are never modified.

        By default, an ``AllocSupport`` instance uses any initiated libEnsemble resource
        module and the built-in libEnsemble scheduler.

        :param W: A :doc:`Worker array<../data_structures/worker_array>`
        :param H: A :doc:`history array<../data_structures/history_array>`
        :param persis_info: Optional, A :doc:`dictionary of persistent information.<../data_structures/libE_specs>`
        :param scheduler_opts: Optional, A dictionary of options to pass to the resource scheduler.
        :param user_resources: Optional, A user supplied ``resources`` object.
        :param user_scheduler: Optional, A user supplied ``user_scheduler`` object.
        """

        self.W = W
        self.H = H
        self.persis_info = persis_info
        self.manage_resources = 'resource_sets' in H.dtype.names
        self.resources = user_resources or Resources.resources
        self.sched = None
        if self.resources is not None:
            wrk_resources = self.resources.resource_manager
            self.sched = user_scheduler or ResourceScheduler(wrk_resources, scheduler_opts)


    def assign_resources(self, rsets_req):
        """Schedule resource sets to a work record if possible.

        Raises ``InsufficientFreeResources`` if the
        required resources are not currently available, or
        ``InsufficientResourcesError`` if the required resources
        do not exist.

        :param rsets_req: Int. Number of resource sets to request.
        :returns: List of Integers. Resource set indices assigned.

        """

        rset_team = None
        if self.resources is not None:
            rset_team = self.sched.assign_resources(rsets_req)
        return rset_team


    # SH TODO: Decision on these functions - Keep as is / make static / init with W (use self.W)
    def avail_worker_ids(self, persistent=None, active_recv=False, zero_resource_workers=None):
        """Returns available workers as a list of IDs, filtered by the given options.

        :param persistent: Optional int. Only return workers with given ``persis_state`` (1 for sim, 2 for gen).
        :param active_recv: Optional Boolean. Only return workers with given active_recv state.
        :param zero_resource_workers: Optional Boolean. If specified, only return workers that require no resources

        If there are no zero resource workers defined, then the ``zero_resource_workers`` argument will
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
        no_zrw = not any(self.W['zero_resource_worker'])
        wrks = []
        for wrk in self.W:
            if fltr_recving() and fltr_persis() and fltr_zrw():
                wrks.append(wrk['worker_id'])
        return wrks


    def count_gens(self):
        """Returns the number of active generators."""
        return sum(self.W['active'] == EVAL_GEN_TAG)


    def test_any_gen(self):
        """Returns ``True`` if a generator worker is active."""
        return any(self.W['active'] == EVAL_GEN_TAG)


    def count_persis_gens(self):
        """Return the number of active persistent generators."""
        return sum(self.W['persis_state'] == EVAL_GEN_TAG)


    def sim_work(self, Work, wid, H_fields, H_rows, persis_info, **libE_info):
        """Add sim work record to given ``Work`` dictionary.

        :param Work: :doc:`Work dictionary<../data_structures/work_dict>`
        :param wid: Worker ID.
        :param H_fields: Which fields from :ref:`H<datastruct-history-array>` to send
        :param H_rows: Which rows of ``H`` to send. Oftentimes, these are non-given, non-cancelled points (``~H['given'] & ~H['cancel_requested']``)
        :param persis_info: Current :ref:`persis_info<datastruct-persis-info>` dictionary

        :returns: None, but ``Work`` is updated.

        Additional passed parameters are inserted into ``libE_info`` in the resulting work record.
        """

        #Should distinguish that it is persis_info for this worker (e.g. wpersis_info ?)
        #why brackets round np.max???

        # Honor rset_team if passed (do we want to check those rsets have been assigned?)
        rset_team = libE_info.get('rset_team', None)

        if self.manage_resources and rset_team is None:
            #import pdb;pdb.set_trace()
            if self.W[wid-1]['persis_state']:
                # IF in persistent state - do not give more resources.
                libE_info['rset_team'] = []
            else:
                num_rsets_req = (np.max(self.H[H_rows]['resource_sets']))
                #print('\nrset_team being called for sim. Requesting {} rsets'.format(num_rsets_req))

                # If more than one group (node) required, finds even split, or allocates whole nodes
                rset_team = self.assign_resources(num_rsets_req)

                # Assign points to worker and remove from task_avail list.
                #print('resource team {} for SIM assigned to worker {}'.format(rset_team, wid), flush=True)
                libE_info['rset_team'] = rset_team

        H_fields = AllocSupport._check_H_fields(H_fields)
        libE_info['H_rows'] = np.atleast_1d(H_rows)

        Work[wid] = {'H_fields': H_fields,
                   'persis_info': persis_info,
                   'tag': EVAL_SIM_TAG,
                   'libE_info': libE_info}

        #print('Work is {}\n'.format(Work[wid]))
        #print('Packed for worker: {}. Resource team for sim: {}\n'.format(wid, rset_team), flush=True)


    # SH TODO: Find/extract commonaility of sim/gen_work.
    def gen_work(self, Work, wid, H_fields, H_rows, persis_info, **libE_info):
        """Add gen work record to given ``Work`` dictionary.

        :param Work: :doc:`Work dictionary<../data_structures/work_dict>`
        :param wid: Worker ID.
        :param H_fields: Which fields from :ref:`H<datastruct-history-array>` to send
        :param H_rows: Which rows of ``H`` to send.
        :param persis_info: Current :ref:`persis_info<datastruct-persis-info>` dictionary

        :returns: None, but ``Work`` is updated.

        Additional passed parameters are inserted into ``libE_info`` in the resulting work record.
        """

        # Honor rset_team if passed (do we want to check those rsets have been assigned?)
        rset_team = libE_info.get('rset_team', None)

        if self.manage_resources and rset_team is None:
            if self.W[wid-1]['persis_state']:
                # IF in persistent state - do not give more resources.
                # persis gen/sims requset for more resources would be dealt with separately.
                # SH TODO: This should be done with sim also - but add when adding persistent sims...
                libE_info['rset_team'] = []
            else:
                # SH TODO: How would you provide resources to a gen? Maybe via persis_info if variable?
                #          Need test where gen_resources is not zero!
                gen_resources = self.persis_info.get('gen_resources', 0)  # This is manager persis_info
                rset_team = self.assign_resources(gen_resources)
                # Even if empty list, presence of non-None rset_team stops manager giving default resources
                libE_info['rset_team'] = rset_team
                #print('resource team {} for GEN assigned to worker {}'.format(rset_team, wid), flush=True)

        # Must come after resources - as that may exit with InsufficientFreeResources
        # SH TODO: Review counter - what if gen work package is cancelled!!!
        AllocSupport.gen_counter += 1  # Count total gens
        libE_info['gen_count'] = AllocSupport.gen_counter

        H_fields = AllocSupport._check_H_fields(H_fields)
        libE_info['H_rows'] = np.atleast_1d(H_rows)

        Work[wid] = {'H_fields': H_fields,
                   'persis_info': persis_info,
                   'tag': EVAL_GEN_TAG,
                   'libE_info': libE_info}


    #def get_points_to_evaluate(self):
        #"""Return points yet to be evaluated"""
        #return ~self.H['given'] & ~self.H['cancel_requested']


    #def get_evaluated_points(self, gen=None):
        #"""Return points that have returned from sim but not yet been given back to gen."""
        #pt_filter = True
        #if gen is not None:
            #gen_inds = (self.H['gen_worker'] == gen)
            #pt_filter = gen_inds
        #return np.logical_and.reduce((self.H['returned'], ~self.H['given_back'], pt_filter))


    #def all_returned(self, gen=None):
        #"""Check if all expected points have returned from sim

        #:param H: A :doc:`history array<../data_structures/history_array>`
        #:param pt_filter: Optional boolean array filtering expected returned points: Default: All True

        #:returns: Boolean. True if all expected points have been returned
        #"""
        #pt_filter = True
        #if gen is not None:
            #gen_inds = (self.H['gen_worker'] == gen)
            #pt_filter = gen_inds

        ## Exclude cancelled points that were not already given out
        #excluded_points = self.H['cancel_requested'] & ~self.H['given']
        #return np.all(self.H['returned'][pt_filter & ~excluded_points])


    def all_returned(self, pt_filter=None, low_bound=None):
        """Returns ``True`` if all expected points have returned from sim

        :param pt_filter: Optional boolean array filtering expected returned points in H.
        :param low_bound: Optional lower bound for testing all returned.
        :returns: True if all expected points have been returned
        """
        # Faster not to slice when whole array
        if low_bound is not None:
            H = self.H[low_bound:]
        else:
            H = self.H

        if pt_filter is None:
            pfilter = True  # Scalar expansion
        else:
            if low_bound is not None:
                pfilter = pt_filter[low_bound:]
            else:
                pfilter = pt_filter

        # Exclude cancelled points that were not already given out
        excluded_points = H['cancel_requested'] & ~H['given']
        return np.all(H['returned'][pfilter & ~excluded_points])


    def points_by_priority(self, points_avail, batch=False):
        """Returns indices of points to give by priority

        :param points_avail: Indices of points that are available to give
        :param batch: Optional Boolean. Should batches of points with the same priority be given simultaneously.
        :returns: An array of point indices to give.
        """

        if 'priority' in self.H.dtype.fields:
            priorities = self.H['priority'][points_avail]
            if batch:
                q_inds = (priorities == np.max(priorities))
            else:
                q_inds = np.argmax(priorities)
        else:
            q_inds = 0
        return np.nonzero(points_avail)[0][q_inds]


    @staticmethod
    def _check_H_fields(H_fields):
        """Ensure no duplicates in H_fields"""
        if len(H_fields) != len(set(H_fields)):
            #logger.debug("Removing duplicate field when packing work request".format(H_fields))
            H_fields = list(set(H_fields))
            #H_fields = list(OrderedDict.fromkeys(H_fields))  # Maintain order
        return H_fields
