import numpy as np
import time
import logging

from libensemble.tools.fields_keys import libE_fields, protected_libE_fields

logger = logging.getLogger(__name__)

# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)


class History:

    """The History class provides methods for managing the history array.

    **Object Attributes:**

    These are set on initialization.

    :ivar numpy_structured_array H:
        History array storing rows for each point. Field names are in
        libensemble/tools/fields_keys.py

    :ivar int offset:
        Starting index for this ensemble (after H0 read in)

    :ivar int index:
        Index where libEnsemble should start filling in H

    :ivar int sim_started_count:
        Number of points given to sim functions (according to H)

    :ivar int sim_ended_count:
        Number of points evaluated  (according to H)

    Note that index, sim_started_count and sim_ended_count reflect the total number of points
    in H and therefore include those prepended to H in addition to the current run.

    """

    # Not currently using libE_specs, persis_info - need to add parameters
    # def __init__(self, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0, persis_info):
    def __init__(self, alloc_specs, sim_specs, gen_specs, exit_criteria, H0):
        """
        Forms the numpy structured array that records everything from the
        libEnsemble run

        """
        L = exit_criteria.get("sim_max", 100)

        # Combine all 'out' fields (if they exist) in sim_specs, gen_specs, or alloc_specs
        specs = [sim_specs, alloc_specs, gen_specs]
        dtype_list = list(set(libE_fields + sum([k.get("out", []) for k in specs if k], [])))
        H = np.zeros(L + len(H0), dtype=dtype_list)  # This may be more history than is needed if H0 has un-given points

        if len(H0):
            # Prepend H with H0
            fields = H0.dtype.names

            for field in fields:
                H[field][: len(H0)] = H0[field]
                # for ind, val in np.ndenumerate(H0[field]):  # Works if H0[field] has arbitrary dimension but is slow
                #     H[field][ind] = val

            if "sim_started" not in fields:
                logger.manager_warning("Marking entries in H0 as having been 'sim_started' and 'sim_ended'")
                H["sim_started"][: len(H0)] = 1
                H["sim_ended"][: len(H0)] = 1
            elif "sim_ended" not in fields:
                logger.manager_warning("Marking entries in H0 as having been 'sim_ended' if 'sim_started'")
                H["sim_ended"][: len(H0)] = H0["sim_started"]

            if "sim_id" not in fields:
                logger.manager_warning("Assigning sim_ids to entries in H0")
                H["sim_id"][: len(H0)] = np.arange(0, len(H0))

        H["sim_id"][-L:] = -1
        H["sim_started_time"][-L:] = np.inf
        H["gen_informed_time"][-L:] = np.inf

        self.H = H
        self.using_H0 = len(H0) > 0
        self.index = len(H0)
        self.grow_count = 0

        self.sim_started_count = np.sum(H["sim_started"])
        self.sim_ended_count = np.sum(H["sim_ended"])
        self.gen_informed_count = np.sum(H["gen_informed"])
        self.given_back_warned = False

        self.sim_started_offset = self.sim_started_count
        self.sim_ended_offset = self.sim_ended_count
        self.gen_informed_offset = self.gen_informed_count

    def update_history_f(self, D, safe_mode):
        """
        Updates the history after points have been evaluated
        """

        new_inds = D["libE_info"]["H_rows"]  # The list of rows (as a numpy array)
        returned_H = D["calc_out"]

        for j, ind in enumerate(new_inds):
            for field in returned_H.dtype.names:
                if safe_mode:
                    assert field not in protected_libE_fields, "The field '" + field + "' is protected"
                if np.isscalar(returned_H[field][j]):
                    self.H[field][ind] = returned_H[field][j]
                else:
                    # len or np.size
                    H0_size = len(returned_H[field][j])
                    assert H0_size <= len(self.H[field][ind]), (
                        "History update Error: Too many values received for " + field
                    )
                    assert H0_size, "History update Error: No values in this field " + field
                    if H0_size == len(self.H[field][ind]):
                        self.H[field][ind] = returned_H[field][j]  # ref
                    else:
                        self.H[field][ind][:H0_size] = returned_H[field][j]  # Slice View

            self.H["sim_ended"][ind] = True
            self.H["sim_ended_time"][ind] = time.time()
            self.sim_ended_count += 1

    def update_history_x_out(self, q_inds, sim_worker):
        """
        Updates the history (in place) when new points have been given out to be evaluated

        Parameters
        ----------
        q_inds: numpy array
            Row IDs for history array H

        sim_worker: integer
            Worker ID
        """
        q_inds = np.atleast_1d(q_inds)
        t = time.time()

        self.H["sim_started"][q_inds] = True
        self.H["sim_started_time"][q_inds] = t
        self.H["sim_worker"][q_inds] = sim_worker

        self.sim_started_count += len(q_inds)

    def update_history_to_gen(self, q_inds):
        """Updates the history (in place) when points are given back to the gen"""
        q_inds = np.atleast_1d(q_inds)
        t = time.time()

        if q_inds.size > 0:
            if np.all(self.H["sim_ended"][q_inds]):
                self.H["gen_informed"][q_inds] = True

            elif np.any(self.H["sim_ended"][q_inds]):  # sporadic returned points need updating
                for ind in q_inds[self.H["sim_ended"][q_inds]]:
                    self.H["gen_informed"][ind] = True

            if self.using_H0 and not self.given_back_warned:
                logger.manager_warning(
                    "Giving entries in H0 back to gen. Marking entries in H0 as 'gen_informed' if 'sim_ended'."
                )
                self.given_back_warned = True

            self.H["gen_informed_time"][q_inds] = t
            self.gen_informed_count += len(q_inds)

    def update_history_x_in(self, gen_worker, D, safe_mode, gen_started_time):
        """
        Updates the history (in place) when new points have been returned from a gen

        Parameters
        ----------
        gen_worker: integer
            The worker who generated these points
        D: numpy array
            Output from gen_func
        """

        if len(D) == 0:
            return

        t = time.time()
        rows_remaining = len(self.H) - self.index

        if "sim_id" not in D.dtype.names:
            # gen method must not be adjusting sim_id, just append to self.H
            num_new = len(D)

            if num_new > rows_remaining:
                self.grow_count = max(num_new - rows_remaining, 2 * self.grow_count)
                self.grow_H(self.grow_count)

            update_inds = np.arange(self.index, self.index + num_new)
            self.H["sim_id"][self.index : self.index + num_new] = range(self.index, self.index + num_new)
        else:
            # gen method is building sim_id or adjusting values in existing sim_id rows.

            # Ensure there aren't any gaps in the generated sim_id values:
            assert np.all(
                np.in1d(np.arange(self.index, np.max(D["sim_id"]) + 1), D["sim_id"])
            ), "The generator function has produced sim_ids that are not in order."

            num_new = len(np.setdiff1d(D["sim_id"], self.H["sim_id"]))

            if num_new > rows_remaining:
                self.grow_count = max(num_new - rows_remaining, 2 * self.grow_count)
                self.grow_H(self.grow_count)

            update_inds = D["sim_id"]

        for field in D.dtype.names:
            if safe_mode:
                assert field not in protected_libE_fields, "The field '" + field + "' is protected"
            self.H[field][update_inds] = D[field]

        first_gen_inds = update_inds[self.H["gen_ended_time"][update_inds] == 0]
        self.H["gen_started_time"][first_gen_inds] = gen_started_time
        self.H["gen_ended_time"][first_gen_inds] = t
        self.H["gen_worker"][first_gen_inds] = gen_worker
        self.index += num_new

    def grow_H(self, k):
        """
        Adds k rows to H in response to gen_f producing more points than
        available rows in H.

        Parameters
        ----------
        k: integer
            Number of rows to add to H
        """
        H_1 = np.zeros(k, dtype=self.H.dtype)
        H_1["sim_id"] = -1
        H_1["sim_started_time"] = np.inf
        H_1["gen_informed_time"] = np.inf
        self.H = np.append(self.H, H_1)

    # Could be arguments here to return different truncations eg. all done, given etc...
    def trim_H(self):
        """Returns truncated array"""
        return self.H[: self.index]
