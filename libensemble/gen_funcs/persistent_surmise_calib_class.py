"""
This module contains a simple calibration example using the Surmise package.
"""

import numpy as np
from surmise.calibration import calibrator
from surmise.emulation import emulator

from libensemble.gen_funcs.surmise_calib_support import (
    gen_observations,
    gen_thetas,
    gen_true_theta,
    gen_xs,
    select_next_theta,
    thetaprior,
)
from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG


def build_emulator(theta, x, fevals):
    """Build the emulator."""
    print(x.shape, theta.shape, fevals.shape)
    emu = emulator(
        x,
        theta,
        fevals,
        method="PCGPwM",
        options={
            "xrmnan": "all",
            "thetarmnan": "never",
            "return_grad": True,
        },
    )
    emu.fit()
    return emu


def select_condition(pending, n_remaining_theta=5):
    n_x = pending.shape[0]
    return False if np.sum(pending) > n_remaining_theta * n_x else True


def rebuild_condition(pending, prev_pending, n_theta=5):  # needs changes
    n_x = pending.shape[0]
    if np.sum(prev_pending) - np.sum(pending) >= n_x * n_theta or np.sum(pending) == 0:
        return True
    else:
        return False


def create_arrays(calc_in, n_thetas, n_x):
    """Create 2D (point * rows) arrays fevals, pending and complete"""
    fevals = np.reshape(calc_in["f"], (n_x, n_thetas))
    pending = np.full(fevals.shape, False)
    prev_pending = pending.copy()
    complete = np.full(fevals.shape, True)

    return fevals, pending, prev_pending, complete


def pad_arrays(n_x, thetanew, theta, fevals, pending, prev_pending, complete):
    """Extend arrays to appropriate sizes."""
    n_thetanew = len(thetanew)

    theta = np.vstack((theta, thetanew))
    fevals = np.hstack((fevals, np.full((n_x, n_thetanew), np.nan)))
    pending = np.hstack((pending, np.full((n_x, n_thetanew), True)))
    prev_pending = np.hstack((prev_pending, np.full((n_x, n_thetanew), True)))
    complete = np.hstack((complete, np.full((n_x, n_thetanew), False)))

    # print('after:', fevals.shape, theta.shape, pending.shape, complete.shape)
    return theta, fevals, pending, prev_pending, complete


def update_arrays(fevals, pending, complete, calc_in, obs_offset, n_x):
    """Unpack from calc_in into 2D (point * rows) fevals"""
    sim_id = calc_in["sim_id"]
    c, r = divmod(sim_id - obs_offset, n_x)  # r, c are arrays if sim_id is an array

    fevals[r, c] = calc_in["f"]
    pending[r, c] = False
    complete[r, c] = True
    return


def cancel_columns_get_H(obs_offset, c, n_x, pending):
    """Cancel columns"""
    sim_ids_to_cancel = []
    columns = np.unique(c)
    for c in columns:
        col_offset = c * n_x
        for i in range(n_x):
            sim_id_cancel = obs_offset + col_offset + i
            if pending[i, c]:
                sim_ids_to_cancel.append(sim_id_cancel)
                pending[i, c] = 0

    H_o = np.zeros(len(sim_ids_to_cancel), dtype=[("sim_id", int), ("cancel_requested", bool)])
    H_o["sim_id"] = sim_ids_to_cancel
    H_o["cancel_requested"] = True
    return H_o


def assign_priority(n_x, n_thetas):
    """Assign priorities to points."""
    # Arbitrary priorities
    priority = np.arange(n_x * n_thetas)
    np.random.shuffle(priority)
    return priority


def load_H(H, xs, thetas, offset=0, set_priorities=False):
    """Fill inputs into H0.

    There will be num_points x num_thetas entries
    """
    n_thetas = len(thetas)
    for i, x in enumerate(xs):
        start = (i + offset) * n_thetas
        H["x"][start : start + n_thetas] = x
        H["thetas"][start : start + n_thetas] = thetas

    if set_priorities:
        n_x = len(xs)
        H["priority"] = assign_priority(n_x, n_thetas)


def gen_truevals(x, gen_specs):
    """Generate true values using libE."""
    n_x = len(x)
    H_o = np.zeros((1) * n_x, dtype=gen_specs["out"])

    # Generate true theta and load into H
    true_theta = gen_true_theta()
    H_o["x"][0:n_x] = x
    H_o["thetas"][0:n_x] = true_theta
    return H_o


class SurmiseCalibrator:
    def __init__(self, persis_info, gen_specs):
        self.gen_specs = gen_specs
        self.rand_stream = persis_info["rand_stream"]
        self.n_thetas = gen_specs["user"]["n_init_thetas"]
        self.n_x = gen_specs["user"]["num_x_vals"]  # Num of x points
        self.step_add_theta = gen_specs["user"]["step_add_theta"]  # No. of thetas to generate per step
        self.n_explore_theta = gen_specs["user"]["n_explore_theta"]  # No. of thetas to explore
        self.obsvar_const = gen_specs["user"]["obsvar"]  # Constant for generator
        self.priorloc = gen_specs["user"]["priorloc"]
        self.priorscale = gen_specs["user"]["priorscale"]
        self.initial_ask = True
        self.initial_tell = True
        self.fevals = None
        self.prev_pending = None

    def ask(self, initial_batch=False, cancellation=False):
        if self.initial_ask:
            self.prior = thetaprior(self.priorloc, self.priorscale)
            self.x = gen_xs(self.n_x, self.rand_stream)
            H_o = gen_truevals(self.x, self.gen_specs)
            self.obs_offset = len(H_o)
            self.initial_ask = False

        elif initial_batch:
            H_o = np.zeros(self.n_x * (self.n_thetas), dtype=self.gen_specs["out"])
            self.theta = gen_thetas(self.prior, self.n_thetas)
            load_H(H_o, self.x, self.theta, set_priorities=True)

        else:
            if select_condition(self.pending):
                new_theta, info = select_next_theta(
                    self.step_add_theta, self.cal, self.emu, self.pending, self.n_explore_theta
                )

                # Add space for new thetas
                self.theta, fevals, pending, self.prev_pending, self.complete = pad_arrays(
                    self.n_x, new_theta, self.theta, self.fevals, self.pending, self.prev_pending, self.complete
                )
                # n_thetas = step_add_theta
                H_o = np.zeros(self.n_x * (len(new_theta)), dtype=self.gen_specs["out"])
                load_H(H_o, self.x, new_theta, set_priorities=True)

                c_obviate = info["obviatesugg"]
                if len(c_obviate) > 0:
                    print(f"columns sent for cancel is:  {c_obviate}", flush=True)
                    H_o = cancel_columns_get_H(self.obs_offset, c_obviate, self.n_x, pending)
                pending[:, c_obviate] = False

        return H_o

    def tell(self, calc_in, tag):
        if self.initial_tell:
            returned_fevals = np.reshape(calc_in["f"], (1, self.n_x))
            true_fevals = returned_fevals
            obs, obsvar = gen_observations(true_fevals, self.obsvar_const, self.rand_stream)
            self.initial_tell = False
            self.ask(initial_batch=True)

        else:
            if self.fevals is None:  # initial batch
                self.fevals, self.pending, prev_pending, self.complete = create_arrays(calc_in, self.n_thetas, self.n_x)
                self.emu = build_emulator(self.theta, self.x, self.fevals)
                # Refer to surmise package for additional options
                self.cal = calibrator(self.emu, obs, self.x, self.prior, obsvar, method="directbayes")

                print("quantiles:", np.round(np.quantile(self.cal.theta.rnd(10000), (0.01, 0.99), axis=0), 3))
                update_model = False
            else:
                # Update fevals from calc_in
                update_arrays(self.fevals, self.pending, self.complete, calc_in, self.obs_offset, self.n_x)
                update_model = rebuild_condition(self.pending, self.prev_pending)
                if not update_model:
                    if tag in [STOP_TAG, PERSIS_STOP]:
                        return

        if update_model:
            print(
                "Percentage Cancelled: %0.2f ( %d / %d)"
                % (
                    100 * np.round(np.mean(1 - self.pending - self.complete), 4),
                    np.sum(1 - self.pending - self.complete),
                    np.prod(self.pending.shape),
                )
            )
            print(
                "Percentage Pending: %0.2f ( %d / %d)"
                % (100 * np.round(np.mean(self.pending), 4), np.sum(self.pending), np.prod(self.pending.shape))
            )
            print(
                "Percentage Complete: %0.2f ( %d / %d)"
                % (100 * np.round(np.mean(self.complete), 4), np.sum(self.complete), np.prod(self.pending.shape))
            )

            self.emu.update(theta=self.theta, f=self.fevals)
            self.cal.fit()

            samples = self.cal.theta.rnd(2500)
            print(np.mean(np.sum((samples - np.array([0.5] * 4)) ** 2, 1)))
            print(np.round(np.quantile(self.cal.theta.rnd(10000), (0.01, 0.99), axis=0), 3))

            self.step_add_theta += 2
            self.prev_pending = self.pending.copy()
            update_model = False

    def finalize(self):
        return None, self.persis_info, FINISHED_PERSISTENT_GEN_TAG
