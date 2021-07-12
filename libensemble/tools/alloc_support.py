import copy
import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.resources.resources import Resources
from libensemble.resources.scheduler import ResourceScheduler

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

class AllocSupport:
    """A helper class to be created/destroyed each time allocation function is called."""

    gen_counter = 0

    # SH TODO: Going to need libE_info passed through to alloc if want to have scheduling options in think
    #          could that way pass through reosurces maybe???? - cos i dont want more class variables being
    #          set. And if want to set scheduler options via libE_specs - have to pass through somehow.
    #          Likely can pass scheduler options in here.
    #          This can hold any caches you want - inc. tasks_avail / lower bounds / avail resource_sets / avail workers etc....
    def __init__(self, user_resources=None, user_scheduler=None):
        self.resources = user_resources or Resources.resources.managerworker_resources

        # If resources is not being used, should spend no time in any resources routines.
        self.sched = None
        #self.rsets_by_group = None
        if self.resources is not None:
            self.sched = user_scheduler or ResourceScheduler(user_resources=self.resources)


    # SH TODO: Naming - assign_resources?/assign_rsets?
    #          There may be various scheduling options.
    #          Terminology - E.g: What do we call the H row/s we are sending to the worker? work_item?
    def assign_resources(self, rsets_req, worker_id):
        """Schedule resource sets to a work item if possible and assign to worker

        This routine assigns the resources given by {rsets_req} and gives to
        worker {worker_id}.

        Returns a list of resource sets ids. A return of None implies
        insufficient resources.
        """

        if self.resources is not None:
            if self.sched is not None: #error handling...
                rset_team = self.sched.assign_resources(rsets_req, worker_id)
        return rset_team


    # SH TODO: Decision - Many of these could be static - but may be
    #          simpler to make them instance functions for consistent calling...
    #          Alternatively - could just give it W in init and use self.W

    # SH TODO: This is another cache being held within alloc - so may internalize from init and keep cache here.
    # currently its effectively static.
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
            # SH TODO 'blocked' condition to be removed.
            if not wrk['blocked'] and fltr_recving() and fltr_persis() and fltr_zrw():
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


    # SH TODO: Need to update for resource sets
    # SH TODO: Variant accepting worker_team - need to test
    #          This may replace sim work as is does the blocking
    def sim_work_with_blocking(self, Work, worker_team, H_fields, H_rows, persis_info, **libE_info):
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


    def gen_work(self, Work, i, H_fields, H_rows, persis_info, **libE_info):
        """Add gen work record to given Work array.

        :param W: :doc:`Worker array<../data_structures/worker_array>`
        :param i: Worker ID.
        :param H_fields: Which fields from H to send
        :param persis_info: current persis_info dictionary

        :returns: None
        """

        # Count total gens
        #try:
            #gen_work.gen_counter += 1
        #except AttributeError:
            #gen_work.gen_counter = 1

        AllocSupport.gen_counter += 1

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
