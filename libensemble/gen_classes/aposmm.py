import copy
from typing import List

import numpy as np
from numpy import typing as npt

from libensemble.generators import LibensembleGenThreadInterfacer
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP
from libensemble.tools import add_unique_random_streams


class APOSMM(LibensembleGenThreadInterfacer):
    """
    Standalone object-oriented APOSMM generator
    """

    def __init__(
        self,
        variables: dict,
        objectives: dict,
        History: npt.NDArray = [],
        persis_info: dict = {},
        gen_specs: dict = {},
        libE_info: dict = {},
        **kwargs
    ) -> None:
        from libensemble.gen_funcs.persistent_aposmm import aposmm

        self.variables = variables
        self.objectives = objectives

        gen_specs["gen_f"] = aposmm

        if self.variables:
            self.n = len(self.variables)  # we'll unpack output x's to correspond with variables
            if not kwargs:
                lb = []
                ub = []
                for v in self.variables.values():
                    if isinstance(v, list) and (isinstance(v[0], int) or isinstance(v[0], float)):
                        # we got a range, append to lb and ub
                        lb.append(v[0])
                        ub.append(v[1])
                kwargs["lb"] = np.array(lb)
                kwargs["ub"] = np.array(ub)

        elif not gen_specs.get("out"):  # gen_specs never especially changes for aposmm even as the problem varies
            self.n = len(kwargs["lb"]) or len(kwargs["ub"])
            gen_specs["out"] = [
                ("x", float, self.n),
                ("x_on_cube", float, self.n),
                ("sim_id", int),
                ("local_min", bool),
                ("local_pt", bool),
            ]
            gen_specs["persis_in"] = ["x", "f", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"]
        if not persis_info:
            persis_info = add_unique_random_streams({}, 2, seed=4321)[1]
        super().__init__(History, persis_info, gen_specs, libE_info, **kwargs)
        if not self.persis_info.get("nworkers"):
            self.persis_info["nworkers"] = gen_specs["user"]["max_active_runs"]  # ??????????
        self.all_local_minima = []
        self._ask_idx = 0
        self._last_ask = None
        self._tell_buf = None
        self._n_buffd_results = 0
        self._n_total_results = 0
        self._told_initial_sample = False

    def _slot_in_data(self, results):
        """Slot in libE_calc_in and trial data into corresponding array fields. *Initial sample only!!*"""
        self._tell_buf["f"][self._n_buffd_results] = results["f"]
        self._tell_buf["x"][self._n_buffd_results] = results["x"]
        self._tell_buf["sim_id"][self._n_buffd_results] = results["sim_id"]
        self._tell_buf["x_on_cube"][self._n_buffd_results] = results["x_on_cube"]
        self._tell_buf["local_pt"][self._n_buffd_results] = results["local_pt"]

    @property
    def _array_size(self):
        """Output array size must match either initial sample or N points to evaluate in parallel."""
        user = self.gen_specs["user"]
        return user["initial_sample_size"] if not self._told_initial_sample else user["max_active_runs"]

    @property
    def _enough_initial_sample(self):
        """We're typically happy with at least 90% of the initial sample, or we've already told the initial sample"""
        return (
            self._n_buffd_results >= self.gen_specs["user"]["initial_sample_size"] - 10
        ) or self._told_initial_sample

    def ask_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if (self._last_ask is None) or (
            self._ask_idx >= len(self._last_ask)
        ):  # haven't been asked yet, or all previously enqueued points have been "asked"
            self._ask_idx = 0
            self._last_ask = super().ask_numpy(num_points)
            if self._last_ask[
                "local_min"
            ].any():  # filter out local minima rows, but they're cached in self.all_local_minima
                min_idxs = self._last_ask["local_min"]
                self.all_local_minima.append(self._last_ask[min_idxs])
                self._last_ask = self._last_ask[~min_idxs]
        if num_points > 0:  # we've been asked for a selection of the last ask
            results = np.copy(
                self._last_ask[self._ask_idx : self._ask_idx + num_points]
            )  # if resetting _last_ask later, results may point to "None"
            self._ask_idx += num_points
            return results
        results = np.copy(self._last_ask)
        self.results = results
        self._last_ask = None
        return results

    def tell_numpy(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        if (results is None and tag == PERSIS_STOP) or len(
            results
        ) == self._array_size:  # told to stop, by final_tell or libE
            self._told_initial_sample = True  # we definitely got an initial sample already if one matches
            super().tell_numpy(results, tag)
            return

        if (
            self._n_buffd_results == 0  # ONLY NEED TO BUFFER RESULTS FOR INITIAL SAMPLE????
        ):  # Optimas prefers to give back chunks of initial_sample. So we buffer them
            self._tell_buf = np.zeros(self._array_size, dtype=self.gen_specs["out"] + [("f", float)])

        if not self._enough_initial_sample:
            self._slot_in_data(np.copy(results))
            self._n_buffd_results += len(results)
        self._n_total_results += len(results)

        if not self._told_initial_sample and self._enough_initial_sample:
            self._tell_buf = self._tell_buf[self._tell_buf["sim_id"] != 0]
            super().tell_numpy(self._tell_buf, tag)
            self._told_initial_sample = True
            self._n_buffd_results = 0

        elif self._told_initial_sample:  # probably libE: given back smaller selection. but from alloc, so its ok?
            super().tell_numpy(results, tag)
            self._n_buffd_results = 0  # dont want to send the same point more than once. slotted in earlier

    def ask_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing entries that have been identified as minima."""
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima
