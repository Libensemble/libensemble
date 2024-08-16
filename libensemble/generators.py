import queue as thread_queue
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from numpy import typing as npt

from libensemble.comms.comms import QComm, QCommThread
from libensemble.executors import Executor
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP
from libensemble.tools.tools import add_unique_random_streams
from libensemble.utils.misc import list_dicts_to_np, np_to_list_dicts

"""
NOTE: These generators, implementations, methods, and subclasses are in BETA, and
      may change in future releases.

      The Generator interface is expected to roughly correspond with CAMPA's standard:
      https://github.com/campa-consortium/generator_standard
"""


class Generator(ABC):
    """

    .. code-block:: python

        from libensemble.specs import GenSpecs
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
        gen_specs = GenSpecs(generator=my_generator, ...)
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
    def ask(self, num_points: Optional[int]) -> List[dict]:
        """
        Request the next set of points to evaluate.
        """

    def ask_updates(self) -> List[npt.NDArray]:
        """
        Request any updates to previous points, e.g. minima discovered, points to cancel.
        """

    def tell(self, results: List[dict]) -> None:
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
    ``ask_numpy/tell_numpy`` methods communicate numpy arrays containing the same data.
    """

    def __init__(
        self, gen_specs: dict = {}, History: npt.NDArray = [], persis_info: dict = {}, libE_info: dict = {}, **kwargs
    ):
        self.gen_specs = gen_specs
        if len(kwargs) > 0:  # so user can specify gen-specific parameters as kwargs to constructor
            self.gen_specs["user"] = kwargs
        if not persis_info:
            self.persis_info = add_unique_random_streams({}, 4, seed=4321)[1]
            self.persis_info["nworkers"] = 4
        else:
            self.persis_info = persis_info

    @abstractmethod
    def ask_numpy(self, num_points: Optional[int] = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""

    @abstractmethod
    def tell_numpy(self, results: npt.NDArray) -> None:
        """Send the results, as a NumPy array, of evaluations to the generator."""

    def ask(self, num_points: Optional[int] = 0) -> List[dict]:
        """Request the next set of points to evaluate."""
        return np_to_list_dicts(self.ask_numpy(num_points))

    def tell(self, results: List[dict]) -> None:
        """Send the results of evaluations to the generator."""
        self.tell_numpy(list_dicts_to_np(results))


class LibensembleGenThreadInterfacer(LibensembleGenerator):
    """Implement ask/tell for traditionally written libEnsemble persistent generator functions.
    Still requires a handful of libEnsemble-specific data-structures on initialization.
    """

    def __init__(
        self, gen_specs: dict, History: npt.NDArray = [], persis_info: dict = {}, libE_info: dict = {}
    ) -> None:
        super().__init__(gen_specs, History, persis_info, libE_info)
        self.gen_f = gen_specs["gen_f"]
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
        new_results = np.zeros(len(results), dtype=self.gen_specs["out"] + [("sim_ended", bool), ("f", float)])
        for field in results.dtype.names:
            new_results[field] = results[field]
        new_results["sim_ended"] = True
        return new_results

    def tell(self, results: List[dict], tag: int = EVAL_GEN_TAG) -> None:
        """Send the results of evaluations to the generator."""
        self.tell_numpy(list_dicts_to_np(results), tag)

    def ask_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if not self.thread.running:
            self.thread.run()
        _, ask_full = self.outbox.get()
        return ask_full["calc_out"]

    def tell_numpy(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
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
        self.tell_numpy(results, PERSIS_STOP)  # conversion happens in tell
        return self.thread.result()