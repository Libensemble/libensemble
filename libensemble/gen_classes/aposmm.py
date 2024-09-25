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
        self, History: npt.NDArray = [], persis_info: dict = {}, gen_specs: dict = {}, libE_info: dict = {}, **kwargs
    ) -> None:
        from libensemble.gen_funcs.persistent_aposmm import aposmm

        gen_specs["gen_f"] = aposmm
        if not gen_specs.get("out"):  # gen_specs never especially changes for aposmm even as the problem varies
            n = len(kwargs["lb"]) or len(kwargs["ub"])
            gen_specs["out"] = [
                ("x", float, n),
                ("x_on_cube", float, n),
                ("sim_id", int),
                ("local_min", bool),
                ("local_pt", bool),
            ]
            gen_specs["persis_in"] = ["x", "x_on_cube", "f", "local_pt", "sim_id", "sim_ended", "local_min"]
        if not persis_info:
            persis_info = add_unique_random_streams({}, 2, seed=4321)[1]
            persis_info["nworkers"] = gen_specs["user"]["max_active_runs"]  # ??????????
        super().__init__(History, persis_info, gen_specs, libE_info, **kwargs)
        self.all_local_minima = []
        self.ask_idx = 0
        self.last_ask = None
        self.tell_buf = None
        self.num_evals = 0
        self._told_initial_sample = False

    def _slot_in_data(self, results):
        """Slot in libE_calc_in and trial data into corresponding array fields."""
        for field in ["f", "x", "x_on_cube", "sim_id", "local_pt"]:
            self.tell_buf[field] = results[field]

    @property
    def _array_size(self):
        """Output array size must match either initial sample or N points to evaluate in parallel."""
        user = self.gen_specs["user"]
        return user["initial_sample_size"] if not self._told_initial_sample else user["max_active_runs"]

    @property
    def _enough_initial_sample(self):
        """We're typically happy with at least 90% of the initial sample."""
        return self.num_evals > int(0.9 * self.gen_specs["user"]["initial_sample_size"])

    @property
    def _enough_subsequent_points(self):
        """But we need to evaluate at least N points, for the N local-optimization processes."""
        return self.num_evals >= self.gen_specs["user"]["max_active_runs"]

    def ask_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if (self.last_ask is None) or (
            self.ask_idx >= len(self.last_ask)
        ):  # haven't been asked yet, or all previously enqueued points have been "asked"
            self.ask_idx = 0
            self.last_ask = super().ask_numpy(num_points)
            if self.last_ask[
                "local_min"
            ].any():  # filter out local minima rows, but they're cached in self.all_local_minima
                min_idxs = self.last_ask["local_min"]
                self.all_local_minima.append(self.last_ask[min_idxs])
                self.last_ask = self.last_ask[~min_idxs]
        if num_points > 0:  # we've been asked for a selection of the last ask
            results = np.copy(
                self.last_ask[self.ask_idx : self.ask_idx + num_points]
            )  # if resetting last_ask later, results may point to "None"
            self.ask_idx += num_points
            return results
        results = np.copy(self.last_ask)
        self.results = results
        self.last_ask = None
        return results

    def tell_numpy(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        if tag == PERSIS_STOP:
            super().tell_numpy(None, tag)
            return
        if self.num_evals == 0:
            self.tell_buf = np.zeros(self._array_size, dtype=self.gen_specs["out"] + [("f", float)])
        self._slot_in_data(results)
        self.num_evals += len(results)
        if not self._told_initial_sample and self._enough_initial_sample:
            super().tell_numpy(self.tell_buf, tag)
            self._told_initial_sample = True
            self.num_evals = 0
        elif self._told_initial_sample and self._enough_subsequent_points:
            super().tell_numpy(self.tell_buf, tag)
            self.num_evals = 0

    def ask_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing entries that have been identified as minima."""
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima
