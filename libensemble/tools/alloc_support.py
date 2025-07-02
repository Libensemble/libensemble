import logging

import numpy as np

from libensemble.message_numbers import EVAL_GEN_TAG, EVAL_SIM_TAG
from libensemble.resources.resources import Resources
from libensemble.resources.scheduler import InsufficientFreeResources, InsufficientResourcesError, ResourceScheduler
from libensemble.utils.misc import extract_H_ranges

logger = logging.getLogger(__name__)
# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)


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

    def __init__(
        self, W, manage_resources=False, persis_info={}, libE_info={}, user_resources=None, user_scheduler=None
    ):
        """Instantiate a new AllocSupport instance

        ``W`` is passed in for convenience on init; it is referenced by the various methods,
        but never modified.

        By default, an ``AllocSupport`` instance uses any initiated libEnsemble resource
        module and the built-in libEnsemble scheduler.

        :param W: A :ref:`Worker array<funcguides-workerarray>`
        :param manage_resources: (Optional) Boolean for if to assign resource sets when creating work units.
        :param persis_info: (Optional) A :ref:`dictionary of persistent information.<datastruct-persis-info>`.
        :param scheduler_opts: (Optional) A dictionary of options to pass to the resource scheduler.
        :param user_resources: (Optional) A user supplied ``resources`` object.
        :param user_scheduler: (Optional) A user supplied ``user_scheduler`` object.
        """
        self.W = W
        self.persis_info = persis_info
        self.manage_resources = manage_resources
        self.resources = user_resources or Resources.resources
        self.sched = None
        self.def_gen_num_procs = libE_info.get("gen_num_procs", 0)
        self.def_gen_num_gpus = libE_info.get("gen_num_gpus", 0)
        if self.resources is not None:
            wrk_resources = self.resources.resource_manager
            scheduler_opts = libE_info.get("scheduler_opts", {})
            self.sched = user_scheduler or ResourceScheduler(wrk_resources, scheduler_opts)

    def assign_resources(self, rsets_req, use_gpus=None, user_params=[]):
        """Schedule resource sets to a work record if possible.

        For default scheduler, if more than one group (node) is required,
        will try to find even split, otherwise allocates whole nodes.

        Raises ``InsufficientFreeResources`` if the required resources are not
        currently available, or ``InsufficientResourcesError`` if the required
        resources do not exist.

        :param rsets_req: Int. Number of resource sets to request.
        :param use_gpus: Bool. Whether to use GPU resource sets.
        :param user_params: List of Integers. User parameters num_procs, num_gpus.
        :returns: List of Integers. Resource set indices assigned.
        """
        rset_team = None
        if self.resources is not None:
            # Try schedule to non-gpu rsets first
            if use_gpus is None:
                try:
                    rset_team = self.sched.assign_resources(rsets_req, use_gpus=False, user_params=user_params)
                    return rset_team
                except (InsufficientFreeResources, InsufficientResourcesError):
                    pass

            rset_team = self.sched.assign_resources(rsets_req, use_gpus, user_params)
        return rset_team

    def avail_worker_ids(self, persistent=None, active_recv=False, zero_resource_workers=None, gen_workers=None):
        """Returns available workers as a list of IDs, filtered by the given options.

        :param persistent: (Optional) Int. Only return workers with given ``persis_state`` (1=sim, 2=gen).
        :param active_recv: (Optional) Boolean. Only return workers with given active_recv state.
        :param zero_resource_workers: (Optional) Boolean. Only return workers that require no resources.
        :param gen_workers: (Optional) Boolean. If True, return gen-only workers. If False, return all other workers.
        :returns: List of worker IDs.

        If there are no zero resource workers defined, then the ``zero_resource_workers`` argument will
        be ignored.
        """

        # For abbrev.
        def fltr_persis():
            if persistent is None:
                return True
            return wrk["persis_state"] == persistent

        def fltr_zrw():
            # If none exist or you did not ask for zrw then return True
            if no_zrw or zero_resource_workers is None:
                return True
            return wrk["zero_resource_worker"] == zero_resource_workers

        def fltr_recving():
            if active_recv:
                return wrk["active_recv"]
            else:
                return wrk["active"] == 0

        def fltr_gen_workers():
            if no_gen_workers or gen_workers is None:
                return True
            return wrk["gen_worker"] == gen_workers

        if active_recv and not persistent:
            raise AllocException("Cannot ask for non-persistent active receive workers")

        # If there are no zero resource workers - then ignore zrw (i.e., use only if they exist)
        no_zrw = not any(self.W["zero_resource_worker"])
        no_gen_workers = not any(self.W["gen_worker"])

        wrks = []
        for wrk in self.W:
            if fltr_recving() and fltr_persis() and fltr_zrw() and fltr_gen_workers():
                wrks.append(wrk["worker_id"])
        return wrks

    def count_gens(self):
        """Returns the number of active generators."""
        return sum((self.W["active"] == EVAL_GEN_TAG))

    def test_any_gen(self):
        """Returns ``True`` if a generator worker is active."""
        return any((self.W["active"] == EVAL_GEN_TAG))

    def count_persis_gens(self):
        """Return the number of active persistent generators."""
        return sum(self.W["persis_state"] == EVAL_GEN_TAG)

    def _req_resources_sim(self, libE_info, user_params, H, H_rows):
        """Determine required resources for a sim work unit"""
        use_gpus = None
        if "resource_sets" in H.dtype.names:
            num_rsets_req = np.max(H[H_rows]["resource_sets"])  # sim rsets
        elif "num_procs" in H.dtype.names:
            procs_per_rset = self.resources.resource_manager.procs_per_rset
            num_rsets_req = AllocSupport._convert_rows_to_rsets(
                libE_info, user_params, H, H_rows, procs_per_rset, "num_procs"
            )
        else:
            num_rsets_req = 1
        if "use_gpus" in H.dtype.names:
            if np.any(H[H_rows]["use_gpus"]):
                use_gpus = True
            else:
                use_gpus = False
        if "num_gpus" in H.dtype.names:
            gpus_per_rset = self.resources.resource_manager.gpus_per_rset
            num_rsets_req_for_gpus = AllocSupport._convert_rows_to_rsets(
                libE_info, user_params, H, H_rows, gpus_per_rset, "num_gpus"
            )
            if num_rsets_req_for_gpus > 0:
                use_gpus = True
            num_rsets_req = max(num_rsets_req, num_rsets_req_for_gpus)
        return num_rsets_req, use_gpus

    def _req_resources_gen(self, libE_info, user_params):
        """Determine required resources for a gen work unit"""
        # We could also have libE_specs defaults (gen_num_procs, gen_num_gpus) - passed by libE_info
        use_gpus = None
        num_rsets_req = self.persis_info.get("gen_resources", 0)
        use_gpus = self.persis_info.get("gen_use_gpus", None)  # can be overwritten below
        if not num_rsets_req:
            gen_nprocs = self.persis_info.get("gen_num_procs", self.def_gen_num_procs)
            if gen_nprocs:
                procs_per_rset = self.resources.resource_manager.procs_per_rset
                num_rsets_req = AllocSupport._convert_to_rsets(
                    libE_info, user_params, procs_per_rset, gen_nprocs, "num_procs"
                )
            gen_ngpus = self.persis_info.get("gen_num_gpus", self.def_gen_num_gpus)
            if gen_ngpus:
                gpus_per_rset = self.resources.resource_manager.gpus_per_rset
                num_rsets_req_for_gpus = AllocSupport._convert_to_rsets(
                    libE_info, user_params, gpus_per_rset, gen_ngpus, "num_gpus"
                )
                if num_rsets_req_for_gpus > 0:
                    use_gpus = True
                num_rsets_req = max(num_rsets_req, num_rsets_req_for_gpus)
        return num_rsets_req, use_gpus

    def _update_rset_team(self, libE_info, wid, H=None, H_rows=None):
        """Add rset_team to libE_info."""
        if self.manage_resources and not libE_info.get("rset_team"):
            num_rsets_req = 0
            if self.W[wid]["persis_state"]:
                # Even if empty list, non-None rset_team stops manager giving default resources
                libE_info["rset_team"] = []
                return
            else:
                user_params = []
                # TODO - can't a gen have these (e.g. if have H0) - or if non-persistent
                if H is not None and H_rows is not None:
                    num_rsets_req, use_gpus = self._req_resources_sim(libE_info, user_params, H, H_rows)
                else:
                    num_rsets_req, use_gpus = self._req_resources_gen(libE_info, user_params)
                libE_info["rset_team"] = self.assign_resources(num_rsets_req, use_gpus, user_params)

    def sim_work(self, wid, H, H_fields, H_rows, persis_info, **libE_info):
        """Add sim work record to given ``Work`` dictionary.

         Includes evaluation of required resources if the worker is not in a
         persistent state.

        :param wid: Int. Worker ID.
        :param H: :ref:`History array<funcguides-history>`. For parsing out requested resource sets.
        :param H_fields: Which fields from :ref:`H<funcguides-history>` to send.
        :param H_rows: Which rows of ``H`` to send.
        :param persis_info: Worker specific :ref:`persis_info<datastruct-persis-info>` dictionary.

        :returns: a Work entry.

        Additional passed parameters are inserted into ``libE_info`` in the resulting work record.

        If ``rset_team`` is passed as an additional parameter, it will be honored, assuming that
        any resource checking has already been done.

        """
        # Parse out resource_sets
        self._update_rset_team(libE_info, wid, H=H, H_rows=H_rows)

        H_fields = AllocSupport._check_H_fields(H_fields)
        libE_info["H_rows"] = AllocSupport._check_H_rows(H_rows)

        work = {
            "H_fields": H_fields,
            "persis_info": persis_info,
            "tag": EVAL_SIM_TAG,
            "libE_info": libE_info,
        }

        logger.debug(f"Alloc func packing SIM work for worker {wid}. Packing sim_ids: {extract_H_ranges(work) or None}")
        return work

    def gen_work(self, wid, H_fields, H_rows, persis_info, **libE_info):
        """Add gen work record to given ``Work`` dictionary.

         Includes evaluation of required resources if the worker is not in a
         persistent state.

        :param Work: :ref:`Work dictionary<funcguides-workdict>`.
        :param wid: Worker ID.
        :param H_fields: Which fields from :ref:`H<funcguides-history>` to send.
        :param H_rows: Which rows of ``H`` to send.
        :param persis_info: Worker specific :ref:`persis_info<datastruct-persis-info>` dictionary.

        :returns: A Work entry.

        Additional passed parameters are inserted into ``libE_info`` in the resulting work record.

        If ``rset_team`` is passed as an additional parameter, it will be honored, and assume that
        any resource checking has already been done. For example, passing ``rset_team=[]``, would
        ensure that no resources are assigned.
        """
        self._update_rset_team(libE_info, wid)

        if not self.W[wid]["persis_state"]:
            AllocSupport.gen_counter += 1  # Count total gens
            libE_info["gen_count"] = AllocSupport.gen_counter

        H_fields = AllocSupport._check_H_fields(H_fields)
        libE_info["H_rows"] = AllocSupport._check_H_rows(H_rows)
        libE_info["batch_size"] = len(self.avail_worker_ids(gen_workers=False))

        work = {
            "H_fields": H_fields,
            "persis_info": persis_info,
            "tag": EVAL_GEN_TAG,
            "libE_info": libE_info,
        }

        logger.debug(f"Alloc func packing GEN work for worker {wid}. Packing sim_ids: {extract_H_ranges(work) or None}")
        return work

    def _filter_points(self, H_in, pt_filter, low_bound):
        """Returns H and pt_filter filtered by lower bound

        :param pt_filter: (Optional) Boolean array filtering expected returned points in ``H``.
        :param low_bound: (Optional) Lower bound for testing all returned.
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
        return H, pfilter

    def all_sim_started(self, H, pt_filter=None, low_bound=None):
        """Returns ``True`` if all expected points have started their sim.

        Excludes cancelled points.

        :param pt_filter: (Optional) Boolean array filtering expected returned points in ``H``.
        :param low_bound: (Optional) Lower bound for testing all returned.
        :returns: True if all expected points have started their sim.
        """
        H, pfilter = self._filter_points(H, pt_filter, low_bound)
        excluded_points = H["cancel_requested"]
        return np.all(H["sim_started"][pfilter & ~excluded_points])

    def all_sim_ended(self, H, pt_filter=None, low_bound=None):
        """Returns ``True`` if all expected points have had their sim_end.

        Excludes cancelled points that were not already sim_started.

        :param pt_filter: (Optional) Boolean array filtering expected returned points in ``H``.
        :param low_bound: (Optional) Lower bound for testing all returned.
        :returns: True if all expected points have had their sim_end.
        """
        H, pfilter = self._filter_points(H, pt_filter, low_bound)
        excluded_points = H["cancel_requested"] & ~H["sim_started"]
        return np.all(H["sim_ended"][pfilter & ~excluded_points])

    def all_gen_informed(self, H, pt_filter=None, low_bound=None):
        """Returns ``True`` if gen has been informed of all expected points.

        Excludes cancelled points that were not already given out.

        :param pt_filter: (Optional) Boolean array filtering expected sim_end points in ``H``.
        :param low_bound: (Optional) Lower bound for testing all returned.
        :returns: True if gen have been informed of all expected points.
        """
        H, pfilter = self._filter_points(H, pt_filter, low_bound)
        excluded_points = H["cancel_requested"] & ~H["sim_started"]
        return np.all(H["gen_informed"][pfilter & ~excluded_points])

    def points_by_priority(self, H, points_avail, batch=False):
        """Returns indices of points to give by priority.

        :param points_avail: Indices of points that are available to give.
        :param batch: (Optional) Boolean. Should batches of points with the same priority be given simultaneously.
        :returns: An array of point indices to give.
        """
        if "priority" in H.dtype.fields:
            priorities = H["priority"][points_avail]
            if batch:
                q_inds = priorities == np.max(priorities)
            else:
                q_inds = np.argmax(priorities)
        else:
            q_inds = 0
        return np.nonzero(points_avail)[0][q_inds]

    def skip_canceled_points(self, H, persis_info):
        """Increments the "next_to_give" field in persis_info to skip any cancelled points"""
        while persis_info["next_to_give"] < len(H) and H[persis_info["next_to_give"]]["cancel_requested"]:
            persis_info["next_to_give"] += 1

        return persis_info

    @staticmethod
    def _check_H_rows(H_rows):
        """Ensure H_rows is a numpy array.  If it is not, then convert if possible,
        else raise an error.

        :returns: ndarray. H_rows
        """
        H_rows = np.atleast_1d(H_rows)  # Makes sure a numpy scalar is an ndarray

        if isinstance(H_rows, np.ndarray):
            return H_rows
        try:
            H_rows = np.fromiter(H_rows, int)
        except Exception:
            raise AllocException(f"H_rows could not be converted to a numpy array. Type {type(H_rows)}")
        return H_rows

    @staticmethod
    def _check_H_fields(H_fields):
        """Ensure no duplicates in H_fields"""
        if len(H_fields) != len(set(H_fields)):
            logger.debug(f"Removing duplicate field(s) when packing work request. {H_fields}")
            H_fields = list(set(H_fields))
            # H_fields = list(OrderedDict.fromkeys(H_fields))  # Maintain order
        return H_fields

    @staticmethod
    def _convert_rows_to_rsets(libE_info, user_params, H, H_rows, units_per_rset, units_str):
        """Convert num_procs & num_gpus requirements to resource sets for sim functions"""
        max_num_units = int(np.max(H[H_rows][units_str]))
        num_rsets_req = AllocSupport._convert_to_rsets(libE_info, user_params, units_per_rset, max_num_units, units_str)
        return num_rsets_req

    @staticmethod
    def _convert_to_rsets(libE_info, user_params, units_per_rset, num_units, units_str):
        """Convert num_procs & num_gpus requirements to resource sets"""
        user_params.append(num_units)
        if num_units > 0:
            try:
                num_rsets_req = num_units // units_per_rset + (num_units % units_per_rset > 0)
            except ZeroDivisionError:
                raise InsufficientResourcesError(
                    f"There are zero {units_str} per resource set (worker). Use fewer workers or more resources"
                )
        else:
            num_rsets_req = 0
        libE_info[units_str] = num_units
        return num_rsets_req
