import copy
import queue as thread_queue
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
from numpy import typing as npt

from libensemble.comms.comms import QComm, QCommThread
from libensemble.executors import Executor
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP
from libensemble.tools import add_unique_random_streams


class Generator(ABC):
    """
    v 0.4.19.24

    Tentative generator interface for use with libEnsemble, and generic enough to be
    broadly compatible with other workflow packages.

    .. code-block:: python

        from libensemble import Ensemble
        from libensemble.generators import Generator


        class MyGenerator(Generator):
            def __init__(self, param):
                self.param = param
                self.model = None

            def ask(self, num_points):
                return create_points(num_points, self.param)

            def tell(self, results):
                self.model = update_model(results, self.model)

            def final_tell(self, results):
                self.tell(results)
                return list(self.model)


        my_generator = MyGenerator(my_parameter=100)
        my_ensemble = Ensemble(generator=my_generator)
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the Generator object on the user-side. Constants, class-attributes,
        and preparation goes here.

        .. code-block:: python

            my_generator = MyGenerator(my_parameter, batch_size=10)
        """

    @abstractmethod
    def ask(self, num_points: Optional[int], *args, **kwargs) -> Iterable:
        """
        Request the next set of points to evaluate, and optionally any previous points to update.
        """

    def ask_updates(self) -> Iterable:
        """
        Request any updates to previous points, e.g. minima discovered, points to cancel.
        """

    def tell(self, results: Iterable, *args, **kwargs) -> None:
        """
        Send the results of evaluations to the generator.
        """

    def final_tell(self, results: Iterable, *args, **kwargs) -> Optional[Iterable]:
        """
        Send the last set of results to the generator, instruct it to cleanup, and
        optionally retrieve an updated final state of evaluations. This is a separate
        method to simplify the common pattern of noting internally if a
        specific tell is the last. This will be called only once.
        """


class LibEnsembleGenInterfacer(Generator):
    """Implement ask/tell for traditionally written libEnsemble persistent generator functions.
    Still requires a handful of libEnsemble-specific data-structures on initialization.
    """

    def __init__(
        self, gen_specs: dict, History: npt.NDArray = [], persis_info: dict = {}, libE_info: dict = {}, **kwargs
    ) -> None:
        self.gen_f = gen_specs["gen_f"]
        self.gen_specs = gen_specs
        self.History = History
        self.persis_info = persis_info
        self.libE_info = libE_info

    def setup(self) -> None:
        self.inbox = thread_queue.Queue()  # sending betweween HERE and gen
        self.outbox = thread_queue.Queue()

        comm = QComm(self.inbox, self.outbox)
        self.libE_info["comm"] = comm  # replacing comm so gen sends HERE instead of manager
        self.libE_info["executor"] = Executor.executor

        self.gen = QCommThread(
            self.gen_f,
            None,
            self.History,
            self.persis_info,
            self.gen_specs,
            self.libE_info,
            user_function=True,
        )  # note that self.gen's inbox/outbox are unused by the underlying gen

    def _set_sim_ended(self, results: npt.NDArray) -> npt.NDArray:
        if "sim_ended" in results.dtype.names:
            results["sim_ended"] = True
        else:
            new_results = np.zeros(len(results), dtype=self.gen_specs["out"] + [("sim_ended", bool), ("f", float)])
            for field in results.dtype.names:
                new_results[field] = results[field]
            new_results["sim_ended"] = True
            results = new_results
        return results

    def ask(self, num_points: Optional[int] = 0, *args, **kwargs) -> npt.NDArray:
        if not self.gen.running:
            self.gen.run()
        _, self.last_ask = self.outbox.get()
        return self.last_ask["calc_out"]

    def ask_updates(self) -> npt.NDArray:
        return self.ask()

    def tell(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        if results is not None:
            results = self._set_sim_ended(results)
            self.inbox.put((tag, {"libE_info": {"H_rows": results["sim_id"], "persistent": True, "executor": None}}))
        else:
            self.inbox.put((tag, None))
        self.inbox.put((0, results))

    def final_tell(self, results: npt.NDArray) -> (npt.NDArray, dict, int):
        self.tell(results, PERSIS_STOP)
        return self.gen.result()

    def create_results_array(
        self, length: int = 0, addtl_fields: list = [("f", float)], empty: bool = False
    ) -> npt.NDArray:
        in_length = len(self.results) if not length else length
        new_results = np.zeros(in_length, dtype=self.gen_specs["out"] + addtl_fields)
        if not empty:
            for field in self.gen_specs["out"]:
                new_results[field[0]] = self.results[field[0]]
        return new_results


class APOSMM(LibEnsembleGenInterfacer):
    """
    Standalone object-oriented APOSMM generator
    """

    def __init__(
        self, gen_specs: dict = {}, History: npt.NDArray = [], persis_info: dict = {}, libE_info: dict = {}, **kwargs
    ) -> None:
        from libensemble.gen_funcs.persistent_aposmm import aposmm

        gen_specs["gen_f"] = aposmm
        if len(kwargs) > 0:
            gen_specs["user"] = kwargs
        if not gen_specs.get("out"):
            n = len(kwargs["lb"]) or len(kwargs["ub"])
            gen_specs["out"] = [
                ("x", float, n),
                ("x_on_cube", float, n),
                ("sim_id", int),
                ("local_min", bool),
                ("local_pt", bool),
            ]
            gen_specs["in"] = ["x", "f", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"]
        if not persis_info:
            persis_info = add_unique_random_streams({}, 4)[1]
            persis_info["nworkers"] = 4
        super().__init__(gen_specs, History, persis_info, libE_info)
        self.all_local_minima = []
        self.cached_ask = None
        self.results_idx = 0
        self.last_ask = None

    def ask(self, *args) -> npt.NDArray:
        if self.last_ask is None:  # haven't been asked yet, or all previously enqueued points have been "asked"
            self.last_ask = super().ask()
            if any(
                self.last_ask["local_min"]
            ):  # filter out local minima rows, but they're cached in self.all_local_minima
                min_idxs = self.last_ask["local_min"]
                self.all_local_minima.append(self.last_ask[min_idxs])
                self.last_ask = self.last_ask[~min_idxs]
        if len(args) and isinstance(args[0], int):  # we've been asked for a selection of the last ask
            num_asked = args[0]
            results = self.last_ask[self.results_idx : self.results_idx + num_asked]
            self.results_idx += num_asked
            if self.results_idx >= len(
                self.last_ask
            ):  # all points have been asked out of the selection. next time around, get new points from aposmm
                self.results_idx = 0
                self.last_ask = None
            return results
        results = copy.deepcopy(self.last_ask)
        self.results = results
        self.last_ask = None
        return results

    def ask_updates(self) -> npt.NDArray:
        minima = copy.deepcopy(self.all_local_minima)
        self.all_local_minima = []
        return minima


class Surmise(LibEnsembleGenInterfacer):
    def __init__(
        self, gen_specs: dict, History: npt.NDArray = [], persis_info: dict = {}, libE_info: dict = {}
    ) -> None:
        from libensemble.gen_funcs.persistent_surmise_calib import surmise_calib

        gen_specs["gen_f"] = surmise_calib
        if ("sim_id", int) not in gen_specs["out"]:
            gen_specs["out"].append(("sim_id", int))
        super().__init__(gen_specs, History, persis_info, libE_info)
        self.sim_id_index = 0
        self.all_cancels = []

    def _add_sim_ids(self, array: npt.NDArray) -> npt.NDArray:
        array["sim_id"] = np.arange(self.sim_id_index, self.sim_id_index + len(array))
        self.sim_id_index += len(array)
        return array

    def ready_to_be_asked(self) -> bool:
        return not self.outbox.empty()

    def ask(self, *args) -> (npt.NDArray, Optional[npt.NDArray]):
        output = super().ask()
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

    def ask_updates(self) -> npt.NDArray:
        cancels = copy.deepcopy(self.all_cancels)
        self.all_cancels = []
        return cancels
