import copy
from typing import List

import numpy as np
from generator_standard.vocs import VOCS
from numpy import typing as npt

from libensemble.generators import PersistentGenInterfacer
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP


class APOSMM(PersistentGenInterfacer):
    """
    Standalone object-oriented APOSMM generator
    """

    def __init__(
        self,
        vocs: VOCS,
        History: npt.NDArray = [],
        persis_info: dict = {},
        gen_specs: dict = {},
        libE_info: dict = {},
        **kwargs,
    ) -> None:
        from libensemble.gen_funcs.persistent_aposmm import aposmm

        self.VOCS = vocs

        gen_specs["gen_f"] = aposmm
        self.n = len(list(self.VOCS.variables.keys()))

        gen_specs["user"] = {}
        gen_specs["user"]["lb"] = np.array([vocs.variables[i].domain[0] for i in vocs.variables])
        gen_specs["user"]["ub"] = np.array([vocs.variables[i].domain[1] for i in vocs.variables])

        if not gen_specs.get("out"):  # gen_specs never especially changes for aposmm even as the problem varies
            gen_specs["out"] = [
                ("x", float, self.n),
                ("x_on_cube", float, self.n),
                ("sim_id", int),
                ("local_min", bool),
                ("local_pt", bool),
            ]
            gen_specs["persis_in"] = ["x", "f", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"]
        super().__init__(vocs, History, persis_info, gen_specs, libE_info, **kwargs)

        if not self.persis_info.get("nworkers"):
            self.persis_info["nworkers"] = kwargs.get("nworkers", gen_specs["user"].get("max_active_runs", 4))
        self.all_local_minima = []
        self._suggest_idx = 0
        self._last_suggest = None
        self._ingest_buf = None
        self._n_buffd_results = 0
        self._told_initial_sample = False

    def _slot_in_data(self, results):
        """Slot in libE_calc_in and trial data into corresponding array fields. *Initial sample only!!*"""
        self._ingest_buf[self._n_buffd_results : self._n_buffd_results + len(results)] = results

    def _enough_initial_sample(self):
        return (
            self._n_buffd_results >= int(self.gen_specs["user"]["initial_sample_size"])
        ) or self._told_initial_sample

    def _ready_to_suggest_genf(self):
        """
        We're presumably ready to be suggested IF:
        - When we're working on the initial sample:
            - We have no _last_suggest cached
            - all points given out have returned AND we've been suggested *at least* as many points as we cached
        - When we're done with the initial sample:
            - we've been suggested *at least* as many points as we cached
        """
        if not self._told_initial_sample and self._last_suggest is not None:
            cond = all([i in self._ingest_buf["sim_id"] for i in self._last_suggest["sim_id"]])
        else:
            cond = True
        return self._last_suggest is None or (cond and (self._suggest_idx >= len(self._last_suggest)))

    def suggest_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if self._ready_to_suggest_genf():
            self._suggest_idx = 0
            self._last_suggest = super().suggest_numpy(num_points)

            if self._last_suggest["local_min"].any():  # filter out local minima rows
                min_idxs = self._last_suggest["local_min"]
                self.all_local_minima.append(self._last_suggest[min_idxs])
                self._last_suggest = self._last_suggest[~min_idxs]

        if num_points > 0:  # we've been suggested for a selection of the last suggest
            results = np.copy(self._last_suggest[self._suggest_idx : self._suggest_idx + num_points])
            self._suggest_idx += num_points

        else:
            results = np.copy(self._last_suggest)
            self._last_suggest = None

        return results

    def ingest_numpy(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        if (results is None and tag == PERSIS_STOP) or self._told_initial_sample:
            super().ingest_numpy(results, tag)
            return

        # Initial sample buffering here:

        if self._n_buffd_results == 0:
            self._ingest_buf = np.zeros(self.gen_specs["user"]["initial_sample_size"], dtype=results.dtype)
            self._ingest_buf["sim_id"] = -1

        if not self._enough_initial_sample():
            self._slot_in_data(np.copy(results))
            self._n_buffd_results += len(results)

        if self._enough_initial_sample():
            super().ingest_numpy(self._ingest_buf, tag)
            self._told_initial_sample = True
            self._n_buffd_results = 0

    def suggest_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing entries that have been identified as minima."""
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima
