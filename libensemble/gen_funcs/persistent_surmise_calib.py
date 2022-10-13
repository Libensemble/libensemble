"""
This module contains a simple calibration example using the Surmise package.
"""

__all__ = ["surmise_calib"]

import numpy as np
from libensemble.gen_funcs.surmise_calib_support import (
    gen_xs,
    gen_thetas,
    gen_observations,
    gen_true_theta,
    thetaprior,
    select_next_theta,
)
from surmise.calibration import calibrator
from surmise.emulation import emulator
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport


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


def cancel_columns(obs_offset, c, n_x, pending, ps):
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

    ps.request_cancel_sim_ids(sim_ids_to_cancel)


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


def surmise_calib(H, persis_info, gen_specs, libE_info):
    """Generator to select and obviate parameters for calibration."""
    rand_stream = persis_info["rand_stream"]
    n_thetas = gen_specs["user"]["n_init_thetas"]
    n_x = gen_specs["user"]["num_x_vals"]  # Num of x points
    step_add_theta = gen_specs["user"]["step_add_theta"]  # No. of thetas to generate per step
    n_explore_theta = gen_specs["user"]["n_explore_theta"]  # No. of thetas to explore
    obsvar_const = gen_specs["user"]["obsvar"]  # Constant for generator
    priorloc = gen_specs["user"]["priorloc"]
    priorscale = gen_specs["user"]["priorscale"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    prior = thetaprior(priorloc, priorscale)

    # Create points at which to evaluate the sim
    x = gen_xs(n_x, rand_stream)

    H_o = gen_truevals(x, gen_specs)
    obs_offset = len(H_o)

    tag, Work, calc_in = ps.send_recv(H_o)
    if tag in [STOP_TAG, PERSIS_STOP]:
        return H, persis_info, FINISHED_PERSISTENT_GEN_TAG

    returned_fevals = np.reshape(calc_in["f"], (1, n_x))
    true_fevals = returned_fevals
    obs, obsvar = gen_observations(true_fevals, obsvar_const, rand_stream)

    # Generate a batch of inputs and load into H
    H_o = np.zeros(n_x * (n_thetas), dtype=gen_specs["out"])
    theta = gen_thetas(prior, n_thetas)
    load_H(H_o, x, theta, set_priorities=True)
    tag, Work, calc_in = ps.send_recv(H_o)
    # -------------------------------------------------------------------------

    fevals = None
    prev_pending = None

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if fevals is None:  # initial batch
            fevals, pending, prev_pending, complete = create_arrays(calc_in, n_thetas, n_x)
            emu = build_emulator(theta, x, fevals)
            # Refer to surmise package for additional options
            cal = calibrator(emu, obs, x, prior, obsvar, method="directbayes")

            print("quantiles:", np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis=0), 3))
            update_model = False
        else:
            # Update fevals from calc_in
            update_arrays(fevals, pending, complete, calc_in, obs_offset, n_x)
            update_model = rebuild_condition(pending, prev_pending)
            if not update_model:
                tag, Work, calc_in = ps.recv()
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break

        if update_model:
            print(
                "Percentage Cancelled: %0.2f ( %d / %d)"
                % (
                    100 * np.round(np.mean(1 - pending - complete), 4),
                    np.sum(1 - pending - complete),
                    np.prod(pending.shape),
                )
            )
            print(
                "Percentage Pending: %0.2f ( %d / %d)"
                % (100 * np.round(np.mean(pending), 4), np.sum(pending), np.prod(pending.shape))
            )
            print(
                "Percentage Complete: %0.2f ( %d / %d)"
                % (100 * np.round(np.mean(complete), 4), np.sum(complete), np.prod(pending.shape))
            )

            emu.update(theta=theta, f=fevals)
            cal.fit()

            samples = cal.theta.rnd(2500)
            print(np.mean(np.sum((samples - np.array([0.5] * 4)) ** 2, 1)))
            print(np.round(np.quantile(cal.theta.rnd(10000), (0.01, 0.99), axis=0), 3))

            step_add_theta += 2
            prev_pending = pending.copy()
            update_model = False

        # Conditionally generate new thetas from model
        if select_condition(pending):
            new_theta, info = select_next_theta(step_add_theta, cal, emu, pending, n_explore_theta)

            # Add space for new thetas
            theta, fevals, pending, prev_pending, complete = pad_arrays(
                n_x, new_theta, theta, fevals, pending, prev_pending, complete
            )

            # n_thetas = step_add_theta
            H_o = np.zeros(n_x * (len(new_theta)), dtype=gen_specs["out"])
            load_H(H_o, x, new_theta, set_priorities=True)
            tag, Work, calc_in = ps.send_recv(H_o)

            # Determine evaluations to cancel
            c_obviate = info["obviatesugg"]
            if len(c_obviate) > 0:
                print(f"columns sent for cancel is:  {c_obviate}", flush=True)
                cancel_columns(obs_offset, c_obviate, n_x, pending, ps)
            pending[:, c_obviate] = False

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
