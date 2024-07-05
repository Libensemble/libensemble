"""Generator class exposing gpCAM functionality"""

import time

import numpy as np
from gpcam import GPOptimizer as GP

from libensemble import Generator

# While there are class / func duplicates - re-use functions.
from libensemble.gen_funcs.persistent_gpCAM import (
    _calculate_grid_distances,
    _eval_var,
    _find_eligible_points,
    _generate_mesh,
    _read_testpoints,
)
from libensemble.generators import list_dicts_to_np, np_to_list_dicts

__all__ = [
    "GP_CAM",
    "GP_CAM_Covar",
]


# Note - batch size is set in wrapper currently - and passed to ask as n_trials.
# To support empty ask(), add batch_size back in here.


# Equivalent to function persistent_gpCAM_ask_tell
class GP_CAM(Generator):
    """
    This generation function constructs a global surrogate of `f` values.

    It is a batched method that produces a first batch uniformly random from
    (lb, ub). On subequent iterations, it calls an optimization method to
    produce the next batch of points. This optimization might be too slow
    (relative to the simulation evaluation time) for some use cases.
    """

    def _initialize_gpcAM(self, user_specs):
        """Extract user params"""
        # self.b = user_specs["batch_size"]
        self.lb = np.array(user_specs["lb"])
        self.ub = np.array(user_specs["ub"])
        self.n = len(self.lb)  # dimension
        assert isinstance(self.n, int), "Dimension must be an integer"
        assert isinstance(self.lb, np.ndarray), "lb must be a numpy array"
        assert isinstance(self.ub, np.ndarray), "ub must be a numpy array"
        self.all_x = np.empty((0, self.n))
        self.all_y = np.empty((0, 1))
        np.random.seed(0)

    def __init__(self, H, persis_info, gen_specs, libE_info=None):
        self.H = H
        self.persis_info = persis_info
        self.gen_specs = gen_specs
        self.libE_info = libE_info

        self.U = self.gen_specs["user"]
        self._initialize_gpcAM(self.U)
        self.my_gp = None
        self.noise = 1e-8  # 1e-12

    def ask(self, n_trials) -> list:
        if self.all_x.shape[0] == 0:
            self.x_new = self.persis_info["rand_stream"].uniform(self.lb, self.ub, (n_trials, self.n))
        else:
            start = time.time()
            self.x_new = self.my_gp.ask(
                bounds=np.column_stack((self.lb, self.ub)),
                n=n_trials,
                pop_size=n_trials,
                max_iter=1,
            )["x"]
            print(f"Ask time:{time.time() - start}")
        H_o = np.zeros(n_trials, dtype=self.gen_specs["out"])
        H_o["x"] = self.x_new
        return np_to_list_dicts(H_o)

    def tell(self, calc_in):
        if calc_in is not None:
            calc_in = list_dicts_to_np(calc_in)
            self.y_new = np.atleast_2d(calc_in["f"]).T
            nan_indices = [i for i, fval in enumerate(self.y_new) if np.isnan(fval)]
            self.x_new = np.delete(self.x_new, nan_indices, axis=0)
            self.y_new = np.delete(self.y_new, nan_indices, axis=0)

            self.all_x = np.vstack((self.all_x, self.x_new))
            self.all_y = np.vstack((self.all_y, self.y_new))

            if self.my_gp is None:
                self.my_gp = GP(self.all_x, self.all_y, noise_variances=self.noise * np.ones(len(self.all_y)))
            else:
                self.my_gp.tell(self.all_x, self.all_y, noise_variances=self.noise * np.ones(len(self.all_y)))
            self.my_gp.train()


class GP_CAM_Covar(GP_CAM):
    """
    This generation function constructs a global surrogate of `f` values.

    It is a batched method that produces a first batch uniformly random from
    (lb, ub) and on following iterations samples the GP posterior covariance
    function to find sample points.
    """

    def __init__(self, H, persis_info, gen_specs, libE_info=None):
        super().__init__(H, persis_info, gen_specs, libE_info)
        self.test_points = _read_testpoints(self.U)
        self.x_for_var = None
        self.var_vals = None
        if self.U.get("use_grid"):
            self.num_points = 10
            self.x_for_var = _generate_mesh(self.lb, self.ub, self.num_points)
            self.r_low_init, self.r_high_init = _calculate_grid_distances(self.lb, self.ub, self.num_points)

    def ask(self, n_trials) -> list:
        if self.all_x.shape[0] == 0:
            x_new = self.persis_info["rand_stream"].uniform(self.lb, self.ub, (n_trials, self.n))
        else:
            if not self.U.get("use_grid"):
                x_new = self.x_for_var[np.argsort(self.var_vals)[-n_trials:]]
            else:
                r_high = self.r_high_init
                r_low = self.r_low_init
                x_new = []
                r_cand = r_high  # Let's start with a large radius and stop when we have batchsize points

                sorted_indices = np.argsort(-self.var_vals)
                while len(x_new) < n_trials:
                    x_new = _find_eligible_points(self.x_for_var, sorted_indices, r_cand, n_trials)
                    if len(x_new) < n_trials:
                        r_high = r_cand
                    r_cand = (r_high + r_low) / 2.0

        self.x_new = x_new
        H_o = np.zeros(n_trials, dtype=self.gen_specs["out"])
        H_o["x"] = self.x_new
        return np_to_list_dicts(H_o)

    def tell(self, calc_in):
        if calc_in is not None:
            super().tell(calc_in)
            if not self.U.get("use_grid"):
                n_trials = len(self.y_new)
                self.x_for_var = self.persis_info["rand_stream"].uniform(self.lb, self.ub, (10 * n_trials, self.n))

            self.var_vals = _eval_var(
                self.my_gp, self.all_x, self.all_y, self.x_for_var, self.test_points, self.persis_info
            )
