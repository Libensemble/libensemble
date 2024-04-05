import queue as thread_queue
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
from numpy import typing as npt

from libensemble.comms.comms import QComm, QCommThread
from libensemble.executors import Executor
from libensemble.gen_funcs.persistent_aposmm import aposmm
from libensemble.gen_funcs.persistent_surmise_calib import surmise_calib
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP
from libensemble.tools import add_unique_random_streams


class Generator(ABC):
    """

    Tentative generator interface for use with libEnsemble, and generic enough to be
    broadly compatible with other workflow packages.

    .. code-block:: python

        from libensemble import Ensemble
        from libensemble.generators import Generator


        class MyGenerator(Generator):
            def __init__(self, param):
                self.param = param
                self.model = None

            def initial_ask(self, num_points, yesterdays_points):
                return create_initial_points(num_points, self.param, yesterdays_points)

            def ask(self, num_points):
                return create_points(num_points, self.param)

            def tell(self, results):
                self.model = update_model(results, self.model)

            def final_tell(self, results):
                self.tell(results)
                return list(self.model)


        my_generator = MyGenerator(my_parameter=100)
        my_ensemble = Ensemble(generator=my_generator)

    Pattern of operations:
    0. User initializes the generator class in their script, provides object to workflow/libEnsemble
    1. Initial ask for points from the generator
    2. Send initial points to workflow for evaluation
    while not instructed to cleanup:
        3. Tell results to generator
        4. Ask generator for subsequent points
        5. Send points to workflow for evaluation. Get results and any cleanup instruction.
    6. Perform final_tell to generator, retrieve any final results/points if any.

    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the Generator object on the user-side. Constants, class-attributes,
        and preparation goes here.

        .. code-block:: python

            my_generator = MyGenerator(my_parameter, batch_size=10)
        """

    def initial_ask(self, num_points: int, previous_results: Optional[Iterable], *args, **kwargs) -> Iterable:
        """
        The initial set of generated points is often produced differently than subsequent sets.
        This is a separate method to simplify the common pattern of noting internally if a
        specific ask was the first. Previous results can be provided to build a foundation
        for the initial sample. This will be called only once.
        """

    @abstractmethod
    def ask(self, num_points: int, *args, **kwargs) -> (Iterable, Optional[Iterable]):
        """
        Request the next set of points to evaluate, and optionally any previous points to update.
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


class LibEnsembleGenTranslator(Generator):
    """Implement ask/tell for traditionally written libEnsemble persistent generator functions.
    Still requires a handful of libEnsemble-specific data-structures on initialization.
    """

    def __init__(
        self, gen_specs: dict, History: npt.NDArray = [], persis_info: dict = {}, libE_info: dict = {}
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

    def initial_ask(self, num_points: int = 0, *args) -> npt.NDArray:
        if not self.gen.running:
            self.gen.run()
        return self.ask(num_points)

    def ask(self, num_points: int = 0) -> (Iterable, Optional[npt.NDArray]):
        _, self.last_ask = self.outbox.get()
        return self.last_ask["calc_out"]

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


class APOSMM(LibEnsembleGenTranslator):
    def __init__(
        self, gen_specs: dict, History: npt.NDArray = [], persis_info: dict = {}, libE_info: dict = {}
    ) -> None:
        gen_specs["gen_f"] = aposmm
        if not persis_info:
            persis_info = add_unique_random_streams({}, 4)[1]
            persis_info["nworkers"] = 4
        super().__init__(gen_specs, History, persis_info, libE_info)

    def initial_ask(self, num_points: int = 0, *args) -> npt.NDArray:
        return super().initial_ask(num_points, args)[0]

    def ask(self, num_points: int = 0) -> (npt.NDArray, npt.NDArray):
        results = super().ask(num_points)
        if any(results["local_min"]):
            minima = results[results["local_min"]]
            results = results[~results["local_min"]]
            return results, minima
        return results, np.empty(0, dtype=self.gen_specs["out"])

    def tell(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        super().tell(results, tag)

    def final_tell(self, results: npt.NDArray) -> (npt.NDArray, dict, int):
        return super().final_tell(results)


class Surmise(LibEnsembleGenTranslator):
    def __init__(
        self, gen_specs: dict, History: npt.NDArray = [], persis_info: dict = {}, libE_info: dict = {}
    ) -> None:
        gen_specs["gen_f"] = surmise_calib
        super().__init__(gen_specs, History, persis_info, libE_info)

    def initial_ask(self, num_points: int = 0, *args) -> npt.NDArray:
        return super().initial_ask(num_points, args)[0]

    def ask(self, num_points: int = 0) -> (npt.NDArray):
        return super().ask(num_points)

    def tell(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        super().tell(results, tag)

    def final_tell(self, results: npt.NDArray) -> (npt.NDArray, dict, int):
        return super().final_tell(results)
