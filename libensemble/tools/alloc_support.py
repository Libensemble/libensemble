import numpy as np
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.resources.resources import Resources
from libensemble.resources.scheduler import ResourceScheduler, InsufficientFreeResources  # noqa: F401


class AllocException(Exception):
    """Raised for any exception in the alloc support"""


class AllocSupport:
    """A helper class to assist with writing allocation functions.

    This class contains methods for common operations like populating work
    units, determining which workers are available, evaluating what values
    need to be distributed to workers, and others.

    Note that since the ``alloc_f`` is called periodically by the Manager, this
    class instance (if used) will be recreated/destroyed on each loop.
    """

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
        self.persis_info = persis_info
        self.manage_resources = sched_opts.get('use_resource_sets', False) 
        self.resources = user_resources or Resources.resources
        self.sched = None
        if self.resources is not None:
            wrk_resources = self.resources.resource_manager
            self.sched = user_scheduler or ResourceScheduler(wrk_resources, scheduler_opts)

    def assign_resources(self, rsets_req):
        """Schedule resource sets to a work record if possible.

        For default scheduler, if more than one group (node) is required,
        will try to find even split, otherwise allocates whole nodes.

        Raises ``InsufficientFreeResources`` if the required resources are not
        currently available, or ``InsufficientResourcesError`` if the required
        resources do not exist.

        :param rsets_req: Int. Number of resource sets to request.
        :returns: List of Integers. Resource set indices assigned.
        """
        rset_team = None
        if self.resources is not None:
            rset_team = self.sched.assign_resources(rsets_req)
        return rset_team

    def avail_worker_ids(self, persistent=None, active_recv=False, zero_resource_workers=None):
        """Returns available workers as a list of IDs, filtered by the given options.

        :param persistent: Optional int. Only return workers with given ``persis_state`` (1=sim, 2=gen).
        :param active_recv: Optional Boolean. Only return workers with given active_recv state.
        :param zero_resource_workers: Optional Boolean. Only return workers that require no resources
        :returns: List of worker IDs

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
                return wrk['active_recv']
            else:
                return not wrk['active']

        if active_recv and not persistent:
            raise AllocException("Cannot ask for non-persistent active receive workers")

        # If there are no zero resource workers - then ignore zrw (i.e. use only if they exist)
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

    def sim_work(self, Work, wid, H_fields, H_rows, persis_info, H=None, **libE_info):
        """Add sim work record to given ``Work`` dictionary.

         Includes evaluation of required resources if the worker is not in a
         persistent state.

        :param Work: :doc:`Work dictionary<../data_structures/work_dict>`
        :param wid: Int. Worker ID.
        :param H_fields: Which fields from :ref:`H<datastruct-history-array>` to send
        :param H_rows: Which rows of ``H`` to send.
        :param persis_info: Worker specific :ref:`persis_info<datastruct-persis-info>` dictionary
        :param H: optional. History array, needed only for resource sets

        :returns: None, but ``Work`` is updated.

        Additional passed parameters are inserted into ``libE_info`` in the resulting work record.

        if rset_team is passed as an additional parameter, it will be honored, assuming that
        any resource checking has already been done.

        """

        # Honor rset_team if passed
        rset_team = libE_info.get('rset_team', None)

        if self.manage_resources and rset_team is None:
            if self.W[wid-1]['persis_state']:
                libE_info['rset_team'] = []
            else:
                num_rsets_req = (np.max(H[H_rows]['resource_sets']))
                rset_team = self.assign_resources(num_rsets_req)
                libE_info['rset_team'] = rset_team
                # print('resource team {} for SIM assigned to worker {}'.format(rset_team, wid), flush=True)

        H_fields = AllocSupport._check_H_fields(H_fields)
        libE_info['H_rows'] = np.atleast_1d(H_rows)

        Work[wid] = {'H_fields': H_fields,
                     'persis_info': persis_info,
                     'tag': EVAL_SIM_TAG,
                     'libE_info': libE_info}

    def gen_work(self, Work, wid, H_fields, H_rows, persis_info, **libE_info):
        """Add gen work record to given ``Work`` dictionary.

         Includes evaluation of required resources if the worker is not in a
         persistent state.

        :param Work: :doc:`Work dictionary<../data_structures/work_dict>`
        :param wid: Worker ID.
        :param H_fields: Which fields from :ref:`H<datastruct-history-array>` to send
        :param H_rows: Which rows of ``H`` to send.
        :param persis_info: Worker specific :ref:`persis_info<datastruct-persis-info>` dictionary

        :returns: None, but ``Work`` is updated.

        Additional passed parameters are inserted into ``libE_info`` in the resulting work record.

        if rset_team is passed as an additional parameter, it will be honored, and assume that
        any resource checking has already been done. For example, passing ``rset_team=[]``, would
        ensure that no resources are assigned.
        """

        # Honor rset_team if passed
        rset_team = libE_info.get('rset_team', None)  # Honor rset_team if passed

        if self.manage_resources and rset_team is None:
            if self.W[wid-1]['persis_state']:
                libE_info['rset_team'] = []
            else:
                gen_resources = self.persis_info.get('gen_resources', 0)
                rset_team = self.assign_resources(gen_resources)
                # Even if empty list, presence of non-None rset_team stops manager giving default resources
                libE_info['rset_team'] = rset_team
                # print('resource team {} for GEN assigned to worker {}'.format(rset_team, wid), flush=True)

        # Must come after resources - as that may exit with InsufficientFreeResources
        if not self.W[wid-1]['persis_state']:
            AllocSupport.gen_counter += 1  # Count total gens
            libE_info['gen_count'] = AllocSupport.gen_counter

        H_fields = AllocSupport._check_H_fields(H_fields)
        libE_info['H_rows'] = np.atleast_1d(H_rows)

        Work[wid] = {'H_fields': H_fields,
                     'persis_info': persis_info,
                     'tag': EVAL_GEN_TAG,
                     'libE_info': libE_info}

    def all_returned(self, H_in, pt_filter=None, low_bound=None):
        """Returns ``True`` if all expected points have returned from sim

        :param pt_filter: Optional boolean array filtering expected returned points in ``H``.
        :param low_bound: Optional lower bound for testing all returned.
        :returns: True if all expected points have been returned
        """
        # Faster not to slice when whole array
        if low_bound is not None:
            H = H_in[low_bound:]
        else:
            H = H_in

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

    def points_by_priority(self, H, points_avail, batch=False):
        """Returns indices of points to give by priority

        :param points_avail: Indices of points that are available to give
        :param batch: Optional Boolean. Should batches of points with the same priority be given simultaneously.
        :returns: An array of point indices to give.
        """
        if 'priority' in H.dtype.fields:
            priorities = H['priority'][points_avail]
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
            # logger.debug("Removing duplicate field when packing work request".format(H_fields))
            H_fields = list(set(H_fields))
            # H_fields = list(OrderedDict.fromkeys(H_fields))  # Maintain order
        return H_fields
