"""
Surmise calibration generator using the gest-api interface.

This module contains a calibration generator that uses the Surmise package
for Bayesian calibration with surrogate model emulation. It supports
selective cancellation of pending simulations as the model evolves.
"""

__all__ = ["SurmiseCalibrator"]

import numpy as np
from gest_api.vocs import VOCS
from numpy import typing as npt
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
from libensemble.generators import LibensembleGenerator

# Generator phases
_PHASE_OBS = 0  # Generating observation points (true theta)
_PHASE_INITIAL = 1  # Generating initial theta batch
_PHASE_MAIN = 2  # Main calibration loop
_PHASE_DONE = 3  # Generator has finished


def _build_emulator(theta, x, fevals):
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


def _select_condition(pending, n_remaining_theta=5):
    """Determine whether enough pending evaluations have returned to select new thetas."""
    n_x = pending.shape[0]
    return False if np.sum(pending) > n_remaining_theta * n_x else True


def _rebuild_condition(pending, prev_pending, n_theta=5):
    """Determine whether enough new results have arrived to rebuild the emulator."""
    n_x = pending.shape[0]
    if np.sum(prev_pending) - np.sum(pending) >= n_x * n_theta or np.sum(pending) == 0:
        return True
    else:
        return False


class SurmiseCalibrator(LibensembleGenerator):
    """
    Bayesian calibration generator using the Surmise package.

    Uses a Gaussian process emulator (PCGPwM method) to build a surrogate
    model and a Bayesian calibrator to select informative parameters.
    Supports cancellation of pending simulations that become unnecessary
    as the model evolves.

    Parameters
    ----------
    vocs : VOCS
        Variable and objective specification. Variables should be split into
        two groups by naming convention: ``x``-prefixed variables are input
        coordinates (e.g. ``x0``, ``x1``, ``x2``) and ``theta``-prefixed
        variables are calibration parameters (e.g. ``theta0``, ``theta1``).
        The objective (e.g. ``f``) is the simulation output.
    n_init_thetas : int
        Number of initial theta parameter sets to evaluate.
    num_x_vals : int
        Number of x-coordinate points to create.
    step_add_theta : int
        Number of new theta parameters to generate per selection step.
    n_explore_theta : int
        Number of theta candidates to explore when selecting the next batch.
    obsvar : float
        Variance constant for generating observation noise.
    priorloc : float
        Location (mean) of the normal prior on theta parameters.
    priorscale : float
        Scale (std) of the normal prior on theta parameters.
    random_seed : int
        Seed for the random number generator.
    """

    def __init__(
        self,
        vocs: VOCS,
        n_init_thetas: int = 15,
        num_x_vals: int = 25,
        step_add_theta: int = 10,
        n_explore_theta: int = 200,
        obsvar: float = 0.1,
        priorloc: float = 1,
        priorscale: float = 0.5,
        random_seed: int = 1,
        *args,
        **kwargs,
    ):
        # Determine x and theta variable names from VOCS before super().__init__
        x_var_names = [v for v in vocs.variables if v.startswith("x")]
        theta_var_names = [v for v in vocs.variables if v.startswith("theta")]

        if not x_var_names:
            raise ValueError("VOCS must contain x-prefixed variables (e.g. x0, x1, x2)")
        if not theta_var_names:
            raise ValueError("VOCS must contain theta-prefixed variables (e.g. theta0, theta1)")

        # Set up variables_mapping so the auto-mapping in LibensembleGenerator
        # doesn't map all variables to a single "x" field. We need separate
        # "x" and "thetas" compound fields in the H-array.
        variables_mapping = kwargs.pop("variables_mapping", {})
        if not variables_mapping:
            variables_mapping = {
                "x": x_var_names,
                "thetas": theta_var_names,
            }

        super().__init__(vocs, *args, variables_mapping=variables_mapping, **kwargs)

        self._n_x_dims = len(x_var_names)
        self._n_theta_dims = len(theta_var_names)

        self.n_init_thetas = n_init_thetas
        self.n_x = num_x_vals
        self.step_add_theta = step_add_theta
        self.n_explore_theta = n_explore_theta
        self.obsvar_const = obsvar
        self.rng = np.random.default_rng(random_seed)

        # Set up prior and generate x-coordinates
        self.prior = thetaprior(priorloc, priorscale, self.rng)
        self.x = gen_xs(self.n_x, self.rng)

        # Output dtype for compound fields
        self._out_dtype = [
            ("x", float, self._n_x_dims),
            ("thetas", float, self._n_theta_dims),
            ("priority", int),
        ]

        # Internal state
        self._phase = _PHASE_OBS
        self._obs_offset = self.n_x  # Number of observation sim_ids before theta evaluations

        # Observation data (populated after obs results arrive)
        self.obs: npt.NDArray | None = None
        self._obsvar: float | None = None

        # Calibration model objects
        self.emu: emulator | None = None
        self.cal: calibrator | None = None

        # Accumulated theta parameters and evaluation tracking arrays
        self.theta: npt.NDArray | None = None
        self.fevals: npt.NDArray | None = None
        self.pending: npt.NDArray | None = None
        self.prev_pending: npt.NDArray | None = None
        self.complete: npt.NDArray | None = None

        # Pending cancellation updates to be returned by suggest_updates()
        self._pending_cancellations: list[npt.NDArray] = []

    def _validate_vocs(self, vocs) -> None:
        pass

    def _make_output(self, xs, thetas, set_priorities=False):
        """Create output array for a batch of (x, theta) pairs.

        The output uses compound fields ``x`` (shape n_x_dims) and ``thetas``
        (shape n_theta_dims), matching the legacy H-array layout.

        Parameters
        ----------
        xs : np.ndarray
            X-coordinate points, shape (n_x, n_x_dims).
        thetas : np.ndarray
            Theta parameters, shape (n_thetas, n_theta_dims).
        set_priorities : bool
            If True, assign randomized priorities.

        Returns
        -------
        H_o : np.ndarray
            Structured array with one row per (x_i, theta_j) pair.
        """
        n_thetas = len(thetas)
        n_x = len(xs)
        n_points = n_x * n_thetas

        H_o = np.zeros(n_points, dtype=self._out_dtype)

        for i, xval in enumerate(xs):
            start = i * n_thetas
            H_o["x"][start : start + n_thetas] = xval
            H_o["thetas"][start : start + n_thetas] = thetas

        if set_priorities:
            priority = np.arange(n_points)
            self.rng.shuffle(priority)
            H_o["priority"] = priority

        return H_o

    def suggest_numpy(self, num_points: int | None = None) -> npt.NDArray:
        """Return the next batch of points to evaluate.

        The generator transitions through three phases:

        1. **Observation phase**: returns ``(1 * n_x)`` points using the true theta.
        2. **Initial batch phase**: returns ``(n_init_thetas * n_x)`` points.
        3. **Main loop phase**: conditionally returns new theta points or an
           empty array when waiting for more results.
        """
        if self._phase == _PHASE_OBS:
            # Generate observation points using the true theta
            true_theta = gen_true_theta()
            H_o = self._make_output(self.x, true_theta)
            return H_o

        elif self._phase == _PHASE_INITIAL:
            # Generate initial batch of thetas
            self.theta = gen_thetas(self.prior, self.n_init_thetas)
            H_o = self._make_output(self.x, self.theta, set_priorities=True)
            return H_o

        elif self._phase == _PHASE_MAIN:
            return self._suggest_main_loop()

        # Done
        return np.zeros(0, dtype=self._out_dtype)

    def _suggest_main_loop(self):
        """Handle suggest logic for the main calibration loop."""
        # Check if we should generate new thetas
        if _select_condition(self.pending):
            new_theta, info = select_next_theta(
                self.step_add_theta, self.cal, self.emu, self.pending, self.n_explore_theta
            )

            # Extend tracking arrays for new thetas
            self._pad_arrays(new_theta)

            # Build output array
            H_o = self._make_output(self.x, new_theta, set_priorities=True)

            # Determine evaluations to cancel
            c_obviate = info["obviatesugg"]
            if len(c_obviate) > 0:
                print(f"columns sent for cancel is:  {c_obviate}", flush=True)
                self._cancel_columns(c_obviate)
            self.pending[:, c_obviate] = False

            return H_o

        # Nothing to suggest yet - return empty array
        return np.zeros(0, dtype=self._out_dtype)

    def ingest_numpy(self, calc_in: npt.NDArray) -> None:
        """Receive evaluated results and update internal state.

        Handles the three phases: observation results, initial batch results,
        and ongoing calibration results.
        """
        if calc_in is None or len(calc_in) == 0:
            return

        if self._phase == _PHASE_OBS:
            # Observation results - construct obs and obsvar
            returned_fevals: npt.NDArray = np.reshape(calc_in["f"], (1, self.n_x))
            true_fevals = returned_fevals
            self.obs, self._obsvar = gen_observations(true_fevals, self.obsvar_const, self.rng)
            self._phase = _PHASE_INITIAL

        elif self._phase == _PHASE_INITIAL:
            # Initial batch results - build emulator and calibrator
            self.fevals = np.reshape(calc_in["f"], (self.n_x, self.n_init_thetas))
            assert self.fevals is not None
            self.pending = np.full(self.fevals.shape, False)
            self.prev_pending = self.pending.copy()
            self.complete = np.full(self.fevals.shape, True)

            self.emu = _build_emulator(self.theta, self.x, self.fevals)
            self.cal = calibrator(self.emu, self.obs, self.x, self.prior, self._obsvar, method="directbayes")
            assert self.cal is not None

            print(
                "quantiles:",
                np.round(np.quantile(self.cal.theta.rnd(10000), (0.01, 0.99), axis=0), 3),
            )
            self._phase = _PHASE_MAIN

        elif self._phase == _PHASE_MAIN:
            # Main loop - update tracking arrays with new results
            self._update_arrays(calc_in)

            # Check if we should rebuild the model
            if _rebuild_condition(self.pending, self.prev_pending):
                self._rebuild_model()

    def _update_arrays(self, calc_in):
        """Unpack results into 2D (n_x, n_thetas) tracking arrays."""
        sim_id = calc_in["sim_id"]
        c, r = divmod(sim_id - self._obs_offset, self.n_x)

        self.fevals[r, c] = calc_in["f"]
        self.pending[r, c] = False
        self.complete[r, c] = True

    def _rebuild_model(self):
        """Rebuild the emulator and recalibrate."""
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
            % (
                100 * np.round(np.mean(self.pending), 4),
                np.sum(self.pending),
                np.prod(self.pending.shape),
            )
        )
        print(
            "Percentage Complete: %0.2f ( %d / %d)"
            % (
                100 * np.round(np.mean(self.complete), 4),
                np.sum(self.complete),
                np.prod(self.pending.shape),
            )
        )

        self.emu.update(theta=self.theta, f=self.fevals)
        self.cal.fit()

        samples = self.cal.theta.rnd(2500)
        print(np.mean(np.sum((samples - np.array([0.5] * self._n_theta_dims)) ** 2, 1)))
        print(np.round(np.quantile(self.cal.theta.rnd(10000), (0.01, 0.99), axis=0), 3))

        self.step_add_theta += 2
        self.prev_pending = self.pending.copy()

    def _pad_arrays(self, new_theta):
        """Extend tracking arrays for new thetas."""
        n_new = len(new_theta)
        self.theta = np.vstack((self.theta, new_theta))
        self.fevals = np.hstack((self.fevals, np.full((self.n_x, n_new), np.nan)))
        self.pending = np.hstack((self.pending, np.full((self.n_x, n_new), True)))
        self.prev_pending = np.hstack((self.prev_pending, np.full((self.n_x, n_new), True)))
        self.complete = np.hstack((self.complete, np.full((self.n_x, n_new), False)))

    def _cancel_columns(self, c_obviate):
        """Mark columns for cancellation and queue cancellation updates."""
        sim_ids_to_cancel = []
        columns = np.unique(c_obviate)
        for c in columns:
            col_offset = c * self.n_x
            for i in range(self.n_x):
                sim_id_cancel = self._obs_offset + col_offset + i
                if self.pending[i, c]:
                    sim_ids_to_cancel.append(sim_id_cancel)
                    self.pending[i, c] = 0

        if sim_ids_to_cancel:
            cancel_array = np.zeros(len(sim_ids_to_cancel), dtype=[("sim_id", int), ("cancel_requested", bool)])
            cancel_array["sim_id"] = sim_ids_to_cancel
            cancel_array["cancel_requested"] = True
            self._pending_cancellations.append(cancel_array)

    def suggest_updates(self):
        """Return pending cancellation updates to be sent to the manager.

        Returns a list of numpy arrays, each containing ``sim_id`` and
        ``cancel_requested`` fields for points that should be cancelled.
        The runner sends these with ``keep_state=True`` so the manager
        updates existing History rows without changing the generator's
        active state.
        """
        updates = self._pending_cancellations
        self._pending_cancellations = []
        return updates
