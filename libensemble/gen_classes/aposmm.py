import copy
from typing import List

import numpy as np
from numpy import typing as npt

from libensemble.generators import LibensembleGenThreadInterfacer
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
            gen_specs["persis_in"] = ["x", "f", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"]
        if not persis_info:
            persis_info = add_unique_random_streams({}, 4, seed=4321)[1]
            persis_info["nworkers"] = 4
        super().__init__(History, persis_info, gen_specs, libE_info, **kwargs)
        self.all_local_minima = []
        self.results_idx = 0
        self.last_ask = None

    def ask_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if (self.last_ask is None) or (
            self.results_idx >= len(self.last_ask)
        ):  # haven't been asked yet, or all previously enqueued points have been "asked"
            self.results_idx = 0
            self.last_ask = super().ask_numpy(num_points)
            if self.last_ask[
                "local_min"
            ].any():  # filter out local minima rows, but they're cached in self.all_local_minima
                min_idxs = self.last_ask["local_min"]
                self.all_local_minima.append(self.last_ask[min_idxs])
                self.last_ask = self.last_ask[~min_idxs]
        if num_points > 0:  # we've been asked for a selection of the last ask
            results = np.copy(
                self.last_ask[self.results_idx : self.results_idx + num_points]
            )  # if resetting last_ask later, results may point to "None"
            self.results_idx += num_points
            return results
        results = np.copy(self.last_ask)
        self.results = results
        self.last_ask = None
        return results

    def ask_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing entries that have been identified as minima."""
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima
