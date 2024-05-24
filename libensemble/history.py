import logging
import time

import numpy as np
import numpy.typing as npt

from libensemble.tools.fields_keys import libE_fields, protected_libE_fields

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, clear_output

logger = logging.getLogger(__name__)

plt.ion()  # Enable interactive mode


# For debug messages - uncomment
# logger.setLevel(logging.DEBUG)

def six_hump_camel_func(x1, x2):
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2
    return term1 + term2 + term3

def init_plot(ax):
    #ax.cla()  # Clear the previous frame
    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-1, 1.1, 50)
    x1, x2 = np.meshgrid(x1, x2)
    f = six_hump_camel_func(x1, x2)
    ax.contourf(x1, x2, f, cmap='winter')  # Change to a contour plot

    #ax.plot_surface(x1, x2, f, rstride=1, cstride=1, cmap='winter', edgecolor='none', alpha=0.6, zorder=1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    #ax.set_zlabel('f')

    #all_pts = ax.scatter3D([], [], [], s=6, color='black', depthshade=False, label='Point')
    #local_pts = ax.scatter3D([], [], [], s=40, color='red', marker='^', zorder=2, depthshade=False, label='Local Point')
    #local_mins = ax.scatter3D([], [], [], s=100, color='yellow', marker='*', zorder=3, depthshade=False, label='Local Min')
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # when not clearing
    #all_pts = ax.scatter([], [], s=6, color='black', zorder=2, alpha=0.5, label='Point')
    #local_pts = ax.scatter([], [], s=40, color='red', marker='^', zorder=3, label='Optimization point')
    #local_mins = ax.scatter([], [], s=100, color='yellow', marker='D', zorder=4, label='Local minimum')


    #ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))


def update_plot(fig, ax, H):
    ax.cla()  # Clear the previous frame
    init_plot(ax)

    # with ax.cla() need to replot - othersie dont - but seeing if makes a difference.


    #SH this should only need ot be done once.
    # Create the surface plot

    ## Extract points from H
    #x1_all = H['x'][:, 0]
    #x2_all = H['x'][:, 1]
    #f_all = H['f']

    #x1_local = H[H['local_pt'] & ~H['local_min']]['x'][:, 0]
    #x2_local = H[H['local_pt'] & ~H['local_min']]['x'][:, 1]
    #f_local = H[H['local_pt'] & ~H['local_min']]['f']

    #x1_min = H[H['local_min']]['x'][:, 0]
    #x2_min = H[H['local_min']]['x'][:, 1]
    #f_min = H[H['local_min']]['f']

    ## Plot the points
    #ax.scatter3D(x1_all, x2_all, f_all, s=50, color='black')
    #ax.scatter3D(x1_min, x2_min, f_min, s=150, color='magenta', marker='*', zorder=3)
    #ax.scatter3D(x1_local, x2_local, f_local, s=15, color='red', marker='^')

    x1 = H['x'][:,0]
    x2 = H['x'][:,1]
    f = H['f']
    #all_pts = ax.scatter3D(x1, x2, f, s=6, color='black', zorder=2, alpha=0.5, label='Point')
    all_pts = ax.scatter(x1, x2, s=6, color='black', zorder=2, alpha=0.5, label='Point')
    all_pts.set_offsets(np.c_[x1, x2])
    #dontwork
    #non_local_min_H = H[~H['local_min']]
    #x1 = non_local_min_H[non_local_min_H['local_pt']]['x'][:, 0]
    #x2 = non_local_min_H[non_local_min_H['local_pt']]['x'][:, 1]
    #f = non_local_min_H[non_local_min_H['local_pt']]['f']
    #local_pts = ax.scatter3D(x1, x2, f, s=15, color='red', marker='^', zorder=2, depthshade=False)

    x1 = H[H['local_pt']]['x'][:,0]
    x2 = H[H['local_pt']]['x'][:,1]
    f = H[H['local_pt']]['f']
    #local_pts = ax.scatter3D(x1, x2, f, s=40, color='red', marker='^', zorder=3, label='Optimization point')
    local_pts = ax.scatter(x1, x2, s=40, color='red', marker='^', zorder=3, label='Optimization point')
    local_pts.set_offsets(np.c_[x1, x2])

    x1 = H[H['local_min']]['x'][:,0]
    x2 = H[H['local_min']]['x'][:,1]
    f = H[H['local_min']]['f']
    #local_mins = ax.scatter3D(x1, x2, f, s=50, color='red', marker='*')
    #local_mins = ax.scatter3D(x1, x2, f, s=100, color='magenta', marker='^')
    #local_mins = ax.scatter3D(x1, x2, f, s=100, color='yellow', marker='D', zorder=4, label='Local minimum')
    local_mins = ax.scatter(x1, x2, s=100, color='yellow', marker='D', zorder=4, label='Local minimum')
    local_mins.set_offsets(np.c_[x1, x2])

    #fig.subplots_adjust(top=0.5, bottom=0.1)
    #plt.title("Points Selected by APOSMM Local Optimization runs", y=1.05)  # Adjust title position
    #plt.title("Points Selected by APOSMM Local Optimization runs", pad=0.5)  # Adjust title position
    #plt.tight_layout(pad=1.0)
    #fig = ax.get_figure()
    fig.tight_layout()
    #fig.subplots_adjust(top=0.8)
    #import pdb;pdb.set_trace()
    plt.title("Points Selected by APOSMM Local Optimization runs", y=1.0)  # Adjust title position
    num_local_min = len(H[H['local_min']])
    #ax.text2D(0.05, 0.8, f'Number of local minima found: {num_local_min}', transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))
    #ax.text2D(0.6, 0.8, f'Points evaluated: {np.sum(H["sim_ended"])}', transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))

    ax.text(0.05, 0.8, f'Number of local minima found: {num_local_min}', transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.6, 0.8, f'Points evaluated: {np.sum(H["sim_ended"])}', transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))

    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))


    display(plt.gcf())
    #display(fig)
    #plt.show()
    #ax.collections.clear()
    clear_output(wait=True)


class History:
    """The History class provides methods for managing the history array.

    **Object Attributes:**

    These are set on initialization.

    :ivar numpy.ndarray H:
        History array storing rows for each point. Field names are in
        libensemble/tools/fields_keys.py. Numpy structured array.

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

    def __init__(
        self, alloc_specs: dict, sim_specs: dict, gen_specs: dict, exit_criteria: dict, H0: npt.NDArray
    ) -> None:
        """
        Forms the numpy structured array that records everything from the
        libEnsemble run

        """
        L = exit_criteria.get("sim_max", 100)

        # Combine all 'out' fields (if they exist) in sim_specs, gen_specs, alloc_specs
        specs = [sim_specs, gen_specs, alloc_specs]
        specs_dtype_list = list(set(libE_fields + sum([k.get("out", []) for k in specs if k], [])))

        if len(H0):
            # remove duplicate fields from specs dtype list if those already in H0 (H0 takes precedence)
            pruned_specs_dtype_list = [i for i in specs_dtype_list if i[0] not in H0.dtype.names]
            H_fields = list(set(pruned_specs_dtype_list + H0.dtype.descr))

            H = np.zeros(L + len(H0), dtype=H_fields)

            # Prepend H with H0
            fields = H0.dtype.names

            for field in fields:
                H[field][: len(H0)] = H0[field]

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
        else:
            H = np.zeros(L + len(H0), dtype=specs_dtype_list)

        H["sim_id"][-L:] = -1
        H["sim_started_time"][-L:] = np.inf
        H["gen_informed_time"][-L:] = np.inf

        if "resource_sets" in H.dtype.names:
            H["resource_sets"][-L:] = 1

        self.H = H
        self.using_H0 = len(H0) > 0
        self.index = len(H0)
        self.grow_count = 0
        self.safe_mode = False

        self.sim_started_count = np.sum(H["sim_started"])
        self.sim_ended_count = np.sum(H["sim_ended"])
        self.gen_informed_count = np.sum(H["gen_informed"])
        self.given_back_warned = False

        self.sim_started_offset = self.sim_started_count
        self.sim_ended_offset = self.sim_ended_count
        self.gen_informed_offset = self.gen_informed_count

        self.last_started = -1
        self.last_ended = -1

        self.last_plot_count = 0

        #self.fig = plt.figure(figsize=(24, 16))
        self.fig = plt.figure(figsize=(7.3, 5.5))
        #self.fig = plt.figure()
        #self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax = self.fig.add_subplot(111)
        init_plot(self.ax)

        #plt.ion()  # Enable interactive mode
        #fig = plt.figure(figsize=(12, 8))
        #ax = fig.add_subplot(111, projection='3d')

    def _append_new_fields(self, H_f: npt.NDArray) -> None:
        dtype_new = np.dtype(list(set(self.H.dtype.descr + H_f.dtype.descr)))
        H_new = np.zeros(len(self.H), dtype=dtype_new)
        old_fields = self.H.dtype.names
        for field in old_fields:
            H_new[field][: len(self.H)] = self.H[field]
        self.H = H_new

    def update_history_f(self, D: dict, kill_canceled_sims: bool = False) -> None:
        """
        Updates the history after points have been evaluated
        """

        new_inds = D["libE_info"]["H_rows"]  # The list of rows (as a numpy array)
        returned_H = D["calc_out"]
        fields = returned_H.dtype.names if returned_H is not None else []

        if returned_H is not None and any([field not in self.H.dtype.names for field in returned_H.dtype.names]):
            self._append_new_fields(returned_H)

        for j, ind in enumerate(new_inds):
            for field in fields:
                if self.safe_mode:
                    assert field not in protected_libE_fields, "The field '" + field + "' is protected"
                if np.isscalar(returned_H[field][j]) or returned_H.dtype[field].hasobject:
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


            update_plot_now = False
            if np.count_nonzero(self.H[self.last_plot_count:]['local_pt']) > 5:
                update_plot_now = True

            if np.any(self.H[self.last_plot_count:]['local_min']):
                update_plot_now = True

            if self.sim_ended_count >= self.last_plot_count + 100:
                update_plot_now = True

            if update_plot_now:
                #plt.ion()  # Enable interactive mode
                #fig = plt.figure(figsize=(24, 16))
                #ax = fig.add_subplot(111, projection='3d')
                update_plot(self.fig, self.ax, self.H)
                self.last_plot_count = self.sim_ended_count
                #plt.ioff()  # Disable interactive mode
                #plt.show()

        if kill_canceled_sims:
            for j in range(self.last_ended + 1, np.max(new_inds) + 1):
                if self.H["sim_ended"][j]:
                    self.last_ended += 1
                else:
                    break

    def update_history_x_out(self, q_inds: npt.NDArray, sim_worker: int, kill_canceled_sims: bool = False) -> None:
        """
        Updates the history (in place) when new points have been given out to be evaluated

        Parameters
        ----------
        q_inds: numpy.typing.NDArray
            Row IDs for history array H

        sim_worker: int
            Worker ID
        """
        q_inds = np.atleast_1d(q_inds)
        t = time.time()

        self.H["sim_started"][q_inds] = True
        self.H["sim_started_time"][q_inds] = t
        self.H["sim_worker"][q_inds] = sim_worker

        self.sim_started_count += len(q_inds)
        if kill_canceled_sims:
            self.last_started = np.max(q_inds)

    def update_history_to_gen(self, q_inds: npt.NDArray):
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

    def update_history_x_in(self, gen_worker: int, D: npt.NDArray, gen_started_time: int) -> None:
        """
        Updates the history (in place) when new points have been returned from a gen

        Parameters
        ----------
        gen_worker: int
            The worker who generated these points
        D: numpy.typing.NDArray
            Output from gen_func
        """

        if len(D) == 0:
            return

        if any([field not in self.H.dtype.names for field in D.dtype.names]):
            self._append_new_fields(D)

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
            if self.safe_mode:
                assert field not in protected_libE_fields, "The field '" + field + "' is protected"
            self.H[field][update_inds] = D[field]

        first_gen_inds = update_inds[self.H["gen_ended_time"][update_inds] == 0]
        self.H["gen_started_time"][first_gen_inds] = gen_started_time
        self.H["gen_ended_time"][first_gen_inds] = t
        self.H["gen_worker"][first_gen_inds] = gen_worker
        self.index += num_new

    def grow_H(self, k: int) -> None:
        """
        Adds k rows to H in response to gen_f producing more points than
        available rows in H.

        Parameters
        ----------
        k: int
            Number of rows to add to H
        """
        H_1 = np.zeros(k, dtype=self.H.dtype)
        H_1["sim_id"] = -1
        H_1["sim_started_time"] = np.inf
        H_1["gen_informed_time"] = np.inf
        if "resource_sets" in H_1.dtype.names:
            H_1["resource_sets"] = 1
        self.H = np.append(self.H, H_1)

    # Could be arguments here to return different truncations eg. all done, given etc...
    def trim_H(self) -> npt.NDArray:
        """Returns truncated array"""
        return self.H[: self.index]
