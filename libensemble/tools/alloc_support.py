import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.resources.resources import Resources
from libensemble.resources.scheduler import ResourceScheduler, InsufficientResourcesException

class AllocException(Exception):
    "Raised for any exception in the alloc support"


class AllocSupport:
    """A helper class to be created/destroyed each time allocation function is called."""

    gen_counter = 0

    #def __init__(self, alloc_specs, manage_resources=False, user_resources=None, user_scheduler=None):
    #def __init__(self, alloc_specs, persis_info, =None, user_resources=None, user_scheduler=None):

    # What if mirror alloc interface (plus resource/scheduler)
    def __init__(self, W, H, sim_specs, gen_specs, alloc_specs, persis_info, user_resources=None, user_scheduler=None):
        """Instantiate a new AllocSupport instance"""

        #self.manage_resources = manage_resources
        self.W = W
        self.H = H
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.alloc_specs = alloc_specs
        self.persis_info = persis_info

        self.manage_resources = 'resource_sets' in H.dtype.names

        self.resources = user_resources or Resources.resources
        self.sched = None
        if self.resources is not None:
            wrk_resources = self.resources.resource_manager
            sched_opts = self.alloc_specs.get('scheduler_opts', {})
            self.sched = user_scheduler or ResourceScheduler(user_resources=wrk_resources, sched_opts=sched_opts)


    def assign_resources(self, rsets_req):
        """Schedule resource sets to a work item if possible.

        Returns a list of resource set ids. A return of None implies
        insufficient resources.
        """
        rset_team = None
        if self.resources is not None:
            rset_team = self.sched.assign_resources(rsets_req)
        return rset_team


    # SH TODO: Decision on these functions - Keep as is / make static / init with W (use self.W)
    def avail_worker_ids(self, persistent=None, active_recv=False, zero_resource_workers=None):
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
        no_zrw = not any(self.W['zero_resource_worker'])
        wrks = []
        for wrk in self.W:
            if fltr_recving() and fltr_persis() and fltr_zrw():
                wrks.append(wrk['worker_id'])
        return wrks


    def count_gens(self):
        """Return the number of active generators in a set of workers.

        :param self.W: :doc:`Worker array<../data_structures/worker_array>`
        """
        return sum(self.W['active'] == EVAL_GEN_TAG)


    def test_any_gen(self):
        """Return True if a generator worker is active.

        :param self.W: :doc:`Worker array<../data_structures/worker_array>`
        """
        return any(self.W['active'] == EVAL_GEN_TAG)


    def count_persis_gens(self):
        """Return the number of active persistent generators in a set of workers.

        :param self.W: :doc:`Worker array<../data_structures/worker_array>`
        """
        return sum(self.W['persis_state'] == EVAL_GEN_TAG)


    def sim_work(self, Work, wid, H_fields, H_rows, persis_info, **libE_info):
        """Add sim work record to given Work array.

        :param self.W: :doc:`Worker array<../data_structures/worker_array>`
        :param wid: Worker ID.
        :param H_fields: Which fields from  to send
        :param persis_info: current persis_info dictionary

        :returns: None
        """

        #Should distinguish that it is persis_info for this worker (e.g. wpersis_info ?)
        #why brackets round np.max???

        # Honor rset_team if passed (do we want to check those rsets have been assigned?)
        rset_team = libE_info.get('rset_team', None)

        if self.manage_resources and rset_team is None:
            #import pdb;pdb.set_trace()
            if self.W[wid-1]['persis_state']:
                # IF in persistent state - do not give more resources.
                rset_team = []
            else:
                num_rsets_req = (np.max(self.H[H_rows]['resource_sets']))
                #print('\nrset_team being called for sim. Requesting {} rsets'.format(num_rsets_req))

                # If more than one group (node) required, finds even split, or allocates whole nodes
                rset_team = self.assign_resources(num_rsets_req)

                # Assign points to worker and remove from task_avail list.
                print('resource team {} for SIM assigned to worker {}'.format(rset_team, wid), flush=True)
                libE_info['rset_team'] = rset_team

        libE_info['H_rows'] = np.atleast_1d(H_rows)
        Work[wid] = {'H_fields': H_fields,
                   'persis_info': persis_info,
                   'tag': EVAL_SIM_TAG,
                   'libE_info': libE_info}

        #print('Work is {}\n'.format(Work[wid]))
        print('Packed for worker: {}. Resource team for sim: {}\n'.format(wid, rset_team), flush=True)


    # SH TODO: Find/extract commonaility of sim/gen_work.
    def gen_work(self, Work, wid, H_fields, H_rows, persis_info, **libE_info):
        """Add gen work record to given Work array.

        :param W: :doc:`Worker array<../data_structures/worker_array>` WRONG ANYWAY
        :param wid: Worker ID.
        :param H_fields: Which fields from  to send
        :param persis_info: current persis_info dictionary

        :returns: None
        """

        # Honor rset_team if passed (do we want to check those rsets have been assigned?)
        rset_team = libE_info.get('rset_team', None)

        if self.manage_resources and rset_team is None:
            if self.W[wid-1]['persis_state']:
                # IF in persistent state - do not give more resources.
                # persis gen/sims requset for more resources would be dealt with separately.
                # SH TODO: This should be done with sim also - but add when adding persistent sims...
                rset_team = []
            else:
                # SH TODO: How would you provide resources to a gen? Maybe via persis_info if variable?
                #          Need test where gen_resources is not zero!
                gen_resources = self.persis_info.get('gen_resources', 0)  # This is manager persis_info
                rset_team = self.assign_resources(gen_resources)
                # Even if empty list, presence of non-None rset_team stops manager giving default resources
                libE_info['rset_team'] = rset_team
                print('resource team {} for GEN assigned to worker {}'.format(rset_team, wid), flush=True)

        # Must come after resources - as that may exit with InsufficientResourcesException
        AllocSupport.gen_counter += 1  # Count total gens
        libE_info['gen_count'] = AllocSupport.gen_counter

        libE_info['H_rows'] = np.atleast_1d(H_rows)
        Work[wid] = {'H_fields': H_fields,
                   'persis_info': persis_info,
                   'tag': EVAL_GEN_TAG,
                   'libE_info': libE_info}


    # SH TODO: Optimization - maybe able to cache gen_inds for gen IDs.
    def get_evaluated_points(self, gen=None):
        """Return points that have been evaluated (returned from sim) but not yet been given back to gen.
        """
        # SH TODO: Will same format as all_returned work?
        #          For first pass keep same as was.
        pt_filter = True
        if gen is not None:
            gen_inds = (self.H['gen_worker'] == gen)
            pt_filter = gen_inds

        gen_inds = (self.H['gen_worker'] == gen)
        return np.logical_and.reduce((self.H['returned'], ~self.H['given_back'], pt_filter))


    def all_returned(self, gen=None):
        """Check if all expected points have returned from sim

        :param H: A :doc:`history array<../data_structures/history_array>`
        :param pt_filter: Optional boolean array filtering expected returned points: Default: All True

        :returns: Boolean. True if all expected points have been returned
        """
        pt_filter = True
        if gen is not None:
            gen_inds = (self.H['gen_worker'] == gen)
            pt_filter = gen_inds

        # Exclude cancelled points that were not already given out
        excluded_points = self.H['cancel_requested'] & ~self.H['given']
        return np.all(self.H['returned'][pt_filter & ~excluded_points])


    #def all_returned(self, pt_filter=True):
        #"""Check if all expected points have returned from sim

        #:param H: A :doc:`history array<../data_structures/history_array>`
        #:param pt_filter: Optional boolean array filtering expected returned points: Default: All True

        #:returns: Boolean. True if all expected points have been returned
        #"""
        ## Exclude cancelled points that were not already given out
        #excluded_points = self.H['cancel_requested'] & ~self.H['given']
        #return np.all(self.H['returned'][pt_filter & ~excluded_points])


    def points_by_priority(self, points_avail, batch=False):
        """Return indices of points to give by priority"""
        if 'priority' in self.H.dtype.fields:
            priorities = self.H['priority'][points_avail]
            if batch:
                q_inds = (priorities == np.max(priorities))
            else:
                q_inds = np.argmax(priorities)
        else:
            q_inds = 0
        return np.nonzero(points_avail)[0][q_inds]