import copy
import queue as thread_queue
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from numpy import typing as npt

from libensemble.comms.comms import QComm, QCommThread
from libensemble.executors import Executor
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP
from libensemble.tools import add_unique_random_streams
from libensemble.utils.misc import list_dicts_to_np, np_to_list_dicts


class Generator(ABC):
    """
    v 0.7.2.24

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
    def ask(self, num_points: Optional[int], *args, **kwargs) -> List[dict]:
        """
        Request the next set of points to evaluate.
        """

    def ask_updates(self) -> npt.NDArray:
        """
        Request any updates to previous points, e.g. minima discovered, points to cancel.
        """

    def tell(self, results: List[dict], *args, **kwargs) -> None:
        """
        Send the results of evaluations to the generator.
        """

    def final_tell(self, results: List[dict], *args, **kwargs) -> Optional[npt.NDArray]:
        """
        Send the last set of results to the generator, instruct it to cleanup, and
        optionally retrieve an updated final state of evaluations. This is a separate
        method to simplify the common pattern of noting internally if a
        specific tell is the last. This will be called only once.
        """


class LibensembleGenerator(Generator):
    """Internal implementation of Generator interface for use with libEnsemble, or for those who
    prefer numpy arrays. ``ask/tell`` methods communicate lists of dictionaries, like the standard.
    ``ask_np/tell_np`` methods communicate numpy arrays containing the same data.
    """

    @abstractmethod
    def ask_np(self, num_points: Optional[int] = 0) -> npt.NDArray:
        pass

    @abstractmethod
    def tell_np(self, results: npt.NDArray) -> None:
        pass

    def ask(self, num_points: Optional[int] = 0) -> List[dict]:
        """Request the next set of points to evaluate."""
        return np_to_list_dicts(self.ask_np(num_points))

    def tell(self, calc_in: List[dict]) -> None:
        """Send the results of evaluations to the generator."""
        self.tell_np(list_dicts_to_np(calc_in))


class LibensembleGenThreadInterfacer(LibensembleGenerator):
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
        self.thread = None

    def setup(self) -> None:
        """Must be called once before calling ask/tell. Initializes the background thread."""
        self.inbox = thread_queue.Queue()  # sending betweween HERE and gen
        self.outbox = thread_queue.Queue()

        comm = QComm(self.inbox, self.outbox)
        self.libE_info["comm"] = comm  # replacing comm so gen sends HERE instead of manager
        self.libE_info["executor"] = Executor.executor

        self.thread = QCommThread(
            self.gen_f,
            None,
            self.History,
            self.persis_info,
            self.gen_specs,
            self.libE_info,
            user_function=True,
        )  # note that self.thread's inbox/outbox are unused by the underlying gen

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

    def tell(self, calc_in: List[dict], tag: int = EVAL_GEN_TAG) -> None:
        """Send the results of evaluations to the generator."""
        self.tell_np(list_dicts_to_np(calc_in), tag)

    def ask_np(self, n_trials: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if not self.thread.running:
            self.thread.run()
        _, ask_full = self.outbox.get()
        return ask_full["calc_out"]

    def ask_updates(self) -> npt.NDArray:
        """Request any updates to previous points, e.g. minima discovered, points to cancel."""
        return self.ask_np()

    def tell_np(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        """Send the results of evaluations to the generator, as a NumPy array."""
        if results is not None:
            results = self._set_sim_ended(results)
            self.inbox.put(
                (tag, {"libE_info": {"H_rows": np.copy(results["sim_id"]), "persistent": True, "executor": None}})
            )
        else:
            self.inbox.put((tag, None))
        self.inbox.put((0, np.copy(results)))

    def final_tell(self, results: npt.NDArray) -> (npt.NDArray, dict, int):
        """Send any last results to the generator, and it to close down."""
        self.tell_np(results, PERSIS_STOP)  # conversion happens in tell
        return self.thread.result()


class APOSMM(LibensembleGenThreadInterfacer):
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
            persis_info = add_unique_random_streams({}, 4, seed=4321)[1]
            persis_info["nworkers"] = 4
        super().__init__(gen_specs, History, persis_info, libE_info)
        self.all_local_minima = []
        self.results_idx = 0
        self.last_ask = None

    def ask_np(self, n_trials: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if (self.last_ask is None) or (
            self.results_idx >= len(self.last_ask)
        ):  # haven't been asked yet, or all previously enqueued points have been "asked"
            self.results_idx = 0
            self.last_ask = super().ask_np(n_trials)
            if self.last_ask[
                "local_min"
            ].any():  # filter out local minima rows, but they're cached in self.all_local_minima
                min_idxs = self.last_ask["local_min"]
                self.all_local_minima.append(self.last_ask[min_idxs])
                self.last_ask = self.last_ask[~min_idxs]
        if n_trials > 0:  # we've been asked for a selection of the last ask
            results = np.copy(
                self.last_ask[self.results_idx : self.results_idx + n_trials]
            )  # if resetting last_ask later, results may point to "None"
            self.results_idx += n_trials
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


class Surmise(LibensembleGenThreadInterfacer):
    """
    Standalone object-oriented Surmise generator
    """

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
        """Check if the generator has the next batch of points ready."""
        return not self.outbox.empty()

    def ask_np(self, *args) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        output = super().ask_np()
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
