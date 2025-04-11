# import queue as thread_queue
from abc import ABC, abstractmethod

# from multiprocessing import Queue as process_queue
from typing import List, Optional

import numpy as np
from numpy import typing as npt

from libensemble.comms.comms import QCommProcess  # , QCommThread
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


class GeneratorNotStartedException(Exception):
    """Exception raised by a threaded/multiprocessed generator upon being suggested without having been started"""


class Generator(ABC):
    """

    .. code-block:: python

        from libensemble.specs import GenSpecs
        from libensemble.generators import Generator


        class MyGenerator(Generator):
            def __init__(self, variables, objectives, param):
                self.param = param
                self.model = create_model(variables, objectives, self.param)

            def suggest(self, num_points):
                return create_points(num_points, self.param)

            def ingest(self, results):
                self.model = update_model(results, self.model)

            def finalize(self, results):
                self.ingest(results)
                return list(self.model)


        variables = {"a": [-1, 1], "b": [-2, 2]}
        objectives = {"f": "MINIMIZE"}

        my_generator = MyGenerator(variables, objectives, my_parameter=100)
        gen_specs = GenSpecs(generator=my_generator, ...)
    """

    @abstractmethod
    def __init__(self, variables: dict[str, List[float]], objectives: dict[str, str], *args, **kwargs):
        """
        Initialize the Generator object on the user-side. Constants, class-attributes,
        and preparation goes here.

        .. code-block:: python

            my_generator = MyGenerator(my_parameter, batch_size=10)
        """

    @abstractmethod
    def suggest(self, num_points: Optional[int]) -> List[dict]:
        """
        Request the next set of points to evaluate.
        """

    def ingest(self, results: List[dict]) -> None:
        """
        Send the results of evaluations to the generator.
        """

    def finalize(self, results: List[dict], *args, **kwargs) -> Optional[npt.NDArray]:
        """
        Send the last set of results to the generator, instruct it to cleanup, and
        optionally retrieve an updated final state of evaluations. This is a separate
        method to simplify the common pattern of noting internally if a
        specific ingest is the last. This will be called only once.
        """


class LibensembleGenerator(Generator):
    """Internal implementation of Generator interface for use with libEnsemble, or for those who
    prefer numpy arrays. ``suggest/ingest`` methods communicate lists of dictionaries, like the standard.
    ``suggest_numpy/ingest_numpy`` methods communicate numpy arrays containing the same data.
    """

    def __init__(
        self,
        variables: dict,
        objectives: dict = {},
        History: npt.NDArray = [],
        persis_info: dict = {},
        gen_specs: dict = {},
        libE_info: dict = {},
        **kwargs,
    ):
        self.variables = variables
        self.objectives = objectives
        self.History = History
        self.gen_specs = gen_specs
        self.libE_info = libE_info

        self.variables_mapping = kwargs.get("variables_mapping", {})

        self._internal_variable = "x"  # need to figure these out dynamically
        self._internal_objective = "f"

        # if self.variables:

        #     self.n = len(self.variables)
        #     # build our own lb and ub
        #     lb = []
        #     ub = []
        #     for i, v in enumerate(self.variables.values()):
        #         if isinstance(v, list) and (isinstance(v[0], int) or isinstance(v[0], float)):
        #             lb.append(v[0])
        #             ub.append(v[1])
        #     kwargs["lb"] = np.array(lb)
        #     kwargs["ub"] = np.array(ub)

        if len(kwargs) > 0:  # so user can specify gen-specific parameters as kwargs to constructor
            if not self.gen_specs.get("user"):
                self.gen_specs["user"] = {}
            self.gen_specs["user"].update(kwargs)
        if not persis_info.get("rand_stream"):
            self.persis_info = add_unique_random_streams({}, 4, seed=4321)[1]
        else:
            self.persis_info = persis_info

    @abstractmethod
    def suggest_numpy(self, num_points: Optional[int] = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""

    @abstractmethod
    def ingest_numpy(self, results: npt.NDArray) -> None:
        """Send the results, as a NumPy array, of evaluations to the generator."""

    @staticmethod
    def convert_np_types(dict_list):
        return [
            {key: (value.item() if isinstance(value, np.generic) else value) for key, value in item.items()}
            for item in dict_list
        ]

    def suggest(self, num_points: Optional[int] = 0) -> List[dict]:
        """Request the next set of points to evaluate."""
        return LibensembleGenerator.convert_np_types(
            np_to_list_dicts(self.suggest_numpy(num_points), mapping=self.variables_mapping)
        )

    def ingest(self, results: List[dict]) -> None:
        """Send the results of evaluations to the generator."""
        self.ingest_numpy(list_dicts_to_np(results, mapping=self.variables_mapping))


class PersistentGenInterfacer(LibensembleGenerator):
    """Implement suggest/ingest for traditionally written libEnsemble persistent generator functions.
    Still requires a handful of libEnsemble-specific data-structures on initialization.
    """

    def __init__(
        self,
        variables: dict,
        objectives: dict = {},
        History: npt.NDArray = [],
        persis_info: dict = {},
        gen_specs: dict = {},
        libE_info: dict = {},
        **kwargs,
    ) -> None:
        super().__init__(variables, objectives, History, persis_info, gen_specs, libE_info, **kwargs)
        self.gen_f = gen_specs["gen_f"]
        self.History = History
        self.libE_info = libE_info
        self.running_gen_f = None

    def setup(self) -> None:
        """Must be called once before calling suggest/ingest. Initializes the background thread."""
        if self.running_gen_f is not None:
            return
        # SH this contains the thread lock -  removing.... wrong comm to pass on anyway.
        if hasattr(Executor.executor, "comm"):
            del Executor.executor.comm
        self.libE_info["executor"] = Executor.executor

        self.running_gen_f = QCommProcess(
            self.gen_f,
            None,
            self.History,
            self.persis_info,
            self.gen_specs,
            self.libE_info,
            user_function=True,
        )

        # this is okay since the object isnt started until the first suggest
        self.libE_info["comm"] = self.running_gen_f.comm

    def _set_sim_ended(self, results: npt.NDArray) -> npt.NDArray:
        filtered_dtype = [
            (name, results.dtype[name]) for name in results.dtype.names if name in self.gen_specs["persis_in"]
        ]
        new_dtype = filtered_dtype + [("sim_ended", bool)]
        new_results = np.zeros(len(results), dtype=new_dtype)
        for field in new_results.dtype.names:  # only copy over fields that generator explicitly wants
            try:
                new_results[field] = results[field]
            except ValueError:  # lets not slot in data that the gen doesnt need?
                continue
        new_results["sim_ended"] = True
        return new_results

    def ingest(self, results: List[dict], tag: int = EVAL_GEN_TAG) -> None:
        """Send the results of evaluations to the generator."""
        self.ingest_numpy(list_dicts_to_np(results, mapping=self.variables_mapping), tag)

    def suggest_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if self.running_gen_f is None:
            self.setup()
            self.running_gen_f.run()
        _, suggest_full = self.running_gen_f.recv()
        return suggest_full["calc_out"]

    def ingest_numpy(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        """Send the results of evaluations to the generator, as a NumPy array."""
        if results is not None:
            results = self._set_sim_ended(results)
            Work = {"libE_info": {"H_rows": np.copy(results["sim_id"]), "persistent": True, "executor": None}}
            self.running_gen_f.send(tag, Work)
            self.running_gen_f.send(
                tag, np.copy(results)
            )  # SH for threads check - might need deepcopy due to dtype=object
        else:
            self.running_gen_f.send(tag, None)

    def finalize(self, results: npt.NDArray = None) -> (npt.NDArray, dict, int):
        """Send any last results to the generator, and it to close down."""
        self.ingest_numpy(results, PERSIS_STOP)  # conversion happens in ingest
        return self.running_gen_f.result()
