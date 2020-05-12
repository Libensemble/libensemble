import numpy as np
import time
import logging

from libensemble.tools.fields_keys import libE_fields

logger = logging.getLogger(__name__)

# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)


class History:

    """The history class provides methods for managing the history array.

    **Object Attributes:**

    These are set on initialization.

    :ivar numpy_structured_array H:
        History array storing rows for each point. Field names are in
        libensemble/tools/fields_keys.py

    :ivar int offset:
        Starting index for this ensemble (after H0 read in)

    :ivar int index:
        Index where libEnsemble should start filling in H

    :ivar int given_count:
        Number of points given to sim fuctions (according to H)

    :ivar int sim_count:
        Number of points evaluated  (according to H)

    Note that index, given_count and sim_count reflect the total number of points
    in H and therefore include those prepended to H in addition to the current run.

    """

    # Not currently using libE_specs, persis_info - need to add parameters
    # def __init__(self, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0, persis_info):
    def __init__(self, alloc_specs, sim_specs, gen_specs, exit_criteria, H0):
        """
        Forms the numpy structured array that records everything from the
        libEnsemble run

        """
        L = exit_criteria.get('sim_max', 100)

        # Combine all 'out' fields (if they exist) in sim_specs, gen_specs, or alloc_specs
        dtype_list = list(set(libE_fields + sum([k['out'] for k in [sim_specs, alloc_specs, gen_specs] if k], [])))
        H = np.zeros(L + len(H0), dtype=dtype_list)

        if len(H0):
            fields = H0.dtype.names

            for field in fields:
                H[field][:len(H0)] = H0[field]
                # for ind, val in np.ndenumerate(H0[field]):  # Works if H0[field] has arbitrary dimension but is slow
                #     H[field][ind] = val

        # Prepend H with H0
        H['sim_id'][:len(H0)] = np.arange(0, len(H0))
        H['given'][:len(H0)] = 1
        H['returned'][:len(H0)] = 1

        H['sim_id'][-L:] = -1
        H['given_time'][-L:] = np.inf

        self.H = H
        # self.offset = 0
        self.offset = len(H0)
        self.index = self.offset

        # libE.check_inputs also checks that all points in H0 are 'returned', so gen and sim have been run.
        # assert np.all(H0['given']), "H0 contains unreturned points. Exiting"
        self.given_count = self.offset

        # assert np.all(H0['returned']), "H0 contains unreturned points. Exiting"
        self.sim_count = self.offset

    def update_history_f(self, D):
        """
        Updates the history after points have been evaluated
        """

        new_inds = D['libE_info']['H_rows']  # The list of rows (as a numpy array)
        returned_H = D['calc_out']

        for j, ind in enumerate(new_inds):
            for field in returned_H.dtype.names:

                if np.isscalar(returned_H[field][j]):
                    self.H[field][ind] = returned_H[field][j]
                else:
                    # len or np.size
                    H0_size = len(returned_H[field][j])
                    assert H0_size <= len(self.H[field][ind]),\
                        "History update Error: Too many values received for " + field
                    assert H0_size, "History update Error: No values in this field " + field
                    if H0_size == len(self.H[field][ind]):
                        self.H[field][ind] = returned_H[field][j]  # ref
                    else:
                        self.H[field][ind][:H0_size] = returned_H[field][j]  # Slice View

            self.H['returned'][ind] = True
            self.sim_count += 1

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
        self.H['given'][q_inds] = True
        self.H['given_time'][q_inds] = time.time()
        self.H['sim_worker'][q_inds] = sim_worker

        if np.isscalar(q_inds):
            self.given_count += 1
        else:
            self.given_count += len(q_inds)

    def update_history_x_in(self, gen_worker, D):
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

        rows_remaining = len(self.H)-self.index

        if 'sim_id' not in D.dtype.names:
            # gen method must not be adjusting sim_id, just append to self.H
            num_new = len(D)

            if num_new > rows_remaining:
                self.grow_H(num_new-rows_remaining)

            update_inds = np.arange(self.index, self.index+num_new)
            self.H['sim_id'][self.index:self.index+num_new] = range(self.index, self.index+num_new)
        else:
            # gen method is building sim_id or adjusting values in existing sim_id rows.

            # Ensure there aren't any gaps in the generated sim_id values:
            assert np.all(np.in1d(np.arange(self.index, np.max(D['sim_id'])+1), D['sim_id'])),\
                "The generator function has produced sim_id that are not in order."

            num_new = len(np.setdiff1d(D['sim_id'], self.H['sim_id']))

            if num_new > rows_remaining:
                self.grow_H(num_new-rows_remaining)

            update_inds = D['sim_id']

        for field in D.dtype.names:
            self.H[field][update_inds] = D[field]

        self.H['gen_time'][update_inds] = time.time()
        self.H['gen_worker'][update_inds] = gen_worker
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
        H_1['sim_id'] = -1
        H_1['given_time'] = np.inf
        self.H = np.append(self.H, H_1)

    # Could be arguments here to return different truncations eg. all done, given etc...
    def trim_H(self):
        """Returns truncated array"""
        return self.H[:self.index]
