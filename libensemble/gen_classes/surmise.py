import copy
import queue as thread_queue
from typing import List

import numpy as np
from numpy import typing as npt

from libensemble.generators import LibensembleGenThreadInterfacer


class Surmise(LibensembleGenThreadInterfacer):
    """
    Standalone object-oriented Surmise generator
    """

    def __init__(
        self, History: npt.NDArray = [], persis_info: dict = {}, gen_specs: dict = {}, libE_info: dict = {}
    ) -> None:
        from libensemble.gen_funcs.persistent_surmise_calib import surmise_calib

        gen_specs["gen_f"] = surmise_calib
        if ("sim_id", int) not in gen_specs["out"]:
            gen_specs["out"].append(("sim_id", int))
        super().__init__(History, persis_info, gen_specs, libE_info)
        self.sim_id_index = 0
        self.all_cancels = []

    def _add_sim_ids(self, array: npt.NDArray) -> npt.NDArray:
        array["sim_id"] = np.arange(self.sim_id_index, self.sim_id_index + len(array))
        self.sim_id_index += len(array)
        return array

    def ready_to_be_asked(self) -> bool:
        """Check if the generator has the next batch of points ready."""
        return not self.outbox.empty()

    def ask_numpy(self, *args) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        output = super().ask_numpy()
        if "cancel_requested" in output.dtype.names:
            cancels = output
            got_cancels_first = True
            self.all_cancels.append(cancels)
        else:
            self.results = self._add_sim_ids(output)
            got_cancels_first = False
        try:
            _, additional = self.outbox.get(timeout=0.2)  # either cancels or new points
            if got_cancels_first:
                return additional["calc_out"]
            self.all_cancels.append(additional["calc_out"])
            return self.results
        except thread_queue.Empty:
            return self.results

    def ask_updates(self) -> List[npt.NDArray]:
        """Request a list of NumPy arrays containing points that should be cancelled by the workflow."""
        cancels = copy.deepcopy(self.all_cancels)
        self.all_cancels = []
        return cancels
