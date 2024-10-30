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
            gen_specs["persis_in"] = ["x", "f", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"]
        if not persis_info:
            persis_info = add_unique_random_streams({}, 2, seed=4321)[1]
        super().__init__(History, persis_info, gen_specs, libE_info, **kwargs)
        if not self.persis_info.get("nworkers"):
            self.persis_info["nworkers"] = kwargs.get("nworkers", gen_specs["user"]["max_active_runs"])
        self.all_local_minima = []
        self._ask_idx = 0
        self._last_ask = None
        self._tell_buf = None
        self._n_buffd_results = 0
        self._n_total_results = 0
        self._told_initial_sample = False

    def _slot_in_data(self, results):
        """Slot in libE_calc_in and trial data into corresponding array fields. *Initial sample only!!*"""
        for field in results.dtype.names:
            self._tell_buf[field][self._n_buffd_results] = results[field]

    def _enough_initial_sample(self):
        return (
            self._n_buffd_results >= int(self.gen_specs["user"]["initial_sample_size"])
        ) or self._told_initial_sample

    def _ready_to_ask_genf(self):
        """We're presumably ready to be asked IF:
        - We have no _last_ask cached
        - the last point given out has returned AND we've been asked *at least* as many points as we cached
        """
        return (
            self._last_ask is None
            or all([i in self._tell_buf["sim_id"] for i in self._last_ask["sim_id"]])
            and (self._ask_idx >= len(self._last_ask))
        )

    def ask_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if self._ready_to_ask_genf():
            self._ask_idx = 0
            self._last_ask = super().ask_numpy(num_points)

            if self._last_ask["local_min"].any():  # filter out local minima rows
                min_idxs = self._last_ask["local_min"]
                self.all_local_minima.append(self._last_ask[min_idxs])
                self._last_ask = self._last_ask[~min_idxs]

        if num_points > 0:  # we've been asked for a selection of the last ask
            results = np.copy(self._last_ask[self._ask_idx : self._ask_idx + num_points])
            self._ask_idx += num_points
            if self._ask_idx >= len(self._last_ask):  # now given out everything; need to reset
                pass  # DEBUGGING WILL CONTINUE HERE

        else:
            results = np.copy(self._last_ask)
            self._last_ask = None

        return results

    def tell_numpy(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        if (results is None and tag == PERSIS_STOP) or self._told_initial_sample:
            super().tell_numpy(results, tag)
            self._n_buffd_results = 0
            return

        # Initial sample buffering here:

        if self._n_buffd_results == 0:
            self._tell_buf = np.zeros(self.gen_specs["user"]["initial_sample_size"], dtype=results.dtype)
            self._tell_buf["sim_id"] = -1

        if not self._enough_initial_sample():
            self._slot_in_data(np.copy(results))
            self._n_buffd_results += len(results)

        if self._enough_initial_sample():
            super().tell_numpy(self._tell_buf, tag)
            self._told_initial_sample = True
            self._n_buffd_results = 0

    def ask_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing entries that have been identified as minima."""
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima
