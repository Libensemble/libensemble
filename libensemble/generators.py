import queue as thread_queue
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from libensemble.comms.comms import QComm, QCommThread
from libensemble.executors import Executor
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP


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

    def initial_ask(self, num_points: int, previous_results: Optional[Iterable]) -> Iterable:
        """
        The initial set of generated points is often produced differently than subsequent sets.
        This is a separate method to simplify the common pattern of noting internally if a
        specific ask was the first. Previous results can be provided to build a foundation
        for the initial sample. This will be called only once.
        """

    @abstractmethod
    def ask(self, num_points: int) -> Iterable:
        """
        Request the next set of points to evaluate.
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

    def __init__(self, gen_f, gen_specs, History=[], persis_info={}, libE_info={}):
        self.gen_f = gen_f
        self.gen_specs = gen_specs
        self.History = History
        self.persis_info = persis_info
        self.libE_info = libE_info

    def init_comms(self):
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

    def initial_ask(self, num_points: int = 0, *args) -> Iterable:
        if not self.gen.running:
            self.gen.run()
        return self.ask(num_points)

    def ask(self, num_points: int = 0) -> Iterable:
        _, self.last_ask = self.outbox.get()
        if num_points:
            return self.last_ask["calc_out"][:num_points]
        return self.last_ask["calc_out"]

    def tell(self, results: Iterable, tag=EVAL_GEN_TAG) -> None:
        if results is not None:
            self.inbox.put((tag, {"libE_info": {"H_rows": results["sim_id"], "persistent": True, "executor": None}}))
        else:
            self.inbox.put((tag, None))
        self.inbox.put((0, results))

    def final_tell(self, results: Iterable) -> Optional[Iterable]:
        self.tell(results, PERSIS_STOP)
        return self.gen.result()
