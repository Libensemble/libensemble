import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.resources.resources import Resources
from libensemble.resources.scheduler import ResourceScheduler, InsufficientResourcesException

class AllocException(Exception):
    "Raised for any exception in the alloc support"


class AllocSupport:
    """A helper class to be created/destroyed each time allocation function is called."""

    gen_counter = 0

    def __init__(self, alloc_specs, user_resources=None, user_scheduler=None):
        """Instantiate a new AllocSupport instance"""
        self.specs = alloc_specs
        self.resources = user_resources or Resources.resources
        self.sched = None
        if self.resources is not None:
            wrk_resources = self.resources.managerworker_resources
            sched_opts = self.specs.get('scheduler_opts', {})
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
    def avail_worker_ids(self, W, persistent=None, active_recv=False, zero_resource_workers=None):
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
            if fltr_recving() and fltr_persis() and fltr_zrw():
                wrks.append(wrk['worker_id'])
        return wrks


    def count_gens(self, W):
        """Return the number of active generators in a set of workers.

        :param W: :doc:`Worker array<../data_structures/worker_array>`
        """
        return sum(W['active'] == EVAL_GEN_TAG)


    def test_any_gen(self, W):
        """Return True if a generator worker is active.

        :param W: :doc:`Worker array<../data_structures/worker_array>`
        """
        return any(W['active'] == EVAL_GEN_TAG)


    def count_persis_gens(self, W):
        """Return the number of active persistent generators in a set of workers.

        :param W: :doc:`Worker array<../data_structures/worker_array>`
        """
        return sum(W['persis_state'] == EVAL_GEN_TAG)


    def sim_work(self, Work, i, H_fields, H_rows, persis_info, **libE_info):
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


    def gen_work(self, Work, i, H_fields, H_rows, persis_info, **libE_info):
        """Add gen work record to given Work array.

        :param W: :doc:`Worker array<../data_structures/worker_array>`
        :param i: Worker ID.
        :param H_fields: Which fields from H to send
        :param persis_info: current persis_info dictionary

        :returns: None
        """

        AllocSupport.gen_counter += 1  # Count total gens
        libE_info['gen_count'] = AllocSupport.gen_counter

        libE_info['H_rows'] = np.atleast_1d(H_rows)
        Work[i] = {'H_fields': H_fields,
                   'persis_info': persis_info,
                   'tag': EVAL_GEN_TAG,
                   'libE_info': libE_info}


    def all_returned(self, H, pt_filter=True):
        """Check if all expected points have returned from sim

        :param H: A :doc:`history array<../data_structures/history_array>`
        :param pt_filter: Optional boolean array filtering expected returned points: Default: All True

        :returns: Boolean. True if all expected points have been returned
        """
        # Exclude cancelled points that were not already given out
        excluded_points = H['cancel_requested'] & ~H['given']
        return np.all(H['returned'][pt_filter & ~excluded_points])
