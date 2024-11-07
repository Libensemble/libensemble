# import queue as thread_queue
from abc import ABC, abstractmethod
from multiprocessing import Manager

# from multiprocessing import Queue as process_queue
from typing import List, Optional

import numpy as np
from numpy import typing as npt

from libensemble.comms.comms import QComm, QCommProcess  # , QCommThread
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
    """Exception raised by a threaded/multiprocessed generator upon being asked without having been started"""


class Generator(ABC):
    """

    .. code-block:: python

        from libensemble.specs import GenSpecs
        from libensemble.generators import Generator


        class MyGenerator(Generator):
            def __init__(self, variables, objectives, param):
                self.param = param
                self.model = create_model(variables, objectives, self.param)

            def ask(self, num_points):
                return create_points(num_points, self.param)

            def tell(self, results):
                self.model = update_model(results, self.model)

            def final_tell(self, results):
                self.tell(results)
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

        if self.variables:

            self.n = len(self.variables)
            # build our own lb and ub
            if "lb" not in kwargs and "ub" not in kwargs:
                lb = []
                ub = []
                for i, v in enumerate(self.variables.values()):
                    if isinstance(v, list) and (isinstance(v[0], int) or isinstance(v[0], float)):
                        lb.append(v[0])
                        ub.append(v[1])
                kwargs["lb"] = np.array(lb)
                kwargs["ub"] = np.array(ub)

        if len(kwargs) > 0:  # so user can specify gen-specific parameters as kwargs to constructor
            if not self.gen_specs.get("user"):
                self.gen_specs["user"] = {}
            self.gen_specs["user"].update(kwargs)
        if not persis_info.get("rand_stream"):
            self.persis_info = add_unique_random_streams({}, 4, seed=4321)[1]
        else:
            self.persis_info = persis_info

    # def _gen_out_to_vars(self, gen_out: dict) -> dict:

    #     """
    #     We must replace internal, enumerated "x"s with the variables the user requested to sample over.

    #     Basically, for the following example, if the user requested the following variables:

    #     ``{'core': [-3, 3], 'edge': [-2, 2]}``

    #     Then for the following directly-from-aposmm point:

    #     ``{'x0': -0.1, 'x1': 0.7, 'x_on_cube0': 0.4833,
    #     'x_on_cube1': 0.675, 'sim_id': 0...}``

    #     We need to replace (for aposmm, for example) "x0" with "core", "x1" with "edge",
    #         "x_on_cube0" with "core_on_cube", and "x_on_cube1" with "edge_on_cube".

    #     ...

    #     BUT: if we're given "x0" and "x1" as our variables, we need to honor that

    #     """

    #     if all([i in list(self.variables.keys()) for i in list(gen_out[0].keys())]):
    #         return gen_out

    #     new_out = []
    #     for entry in gen_out:  # get a dict

    #         new_entry = {}
    #         for map_key in self._vars_x_mapping.keys():  # get 0, 1

    #             for out_key in entry.keys():  # get x0, x1, x_on_cube0, etc.

    #                 if out_key.endswith(str(map_key)):  # found key that ends with 0, 1
    #                     new_name = str(out_key).replace(
    #                         self._internal_variable, self._vars_x_mapping[map_key]
    #                     )  # replace 'x' with 'core'
    #                     new_name = new_name.rstrip("0123456789")  # now remove trailing integer
    #                     new_entry[new_name] = entry[out_key]

    #                 elif not out_key[-1].isnumeric():  # found key that is not enumerated
    #                     new_entry[out_key] = entry[out_key]

    #                 # we now naturally continue over cases where e.g. the map_key may be 0 but we're looking at x1
    #         new_out.append(new_entry)

    #     return new_out

    # def _objs_and_vars_to_gen_in(self, results: dict) -> dict:
    #     """We now need to do the inverse of _gen_out_to_vars, plus replace
    #     the objective name with the internal gen's expected name, .e.g "energy" -> "f".

    #     So given:

    #     {'core': -0.1, 'core_on_cube': 0.483, 'sim_id': 0, 'local_min': False,
    #     'local_pt': False, 'edge': 0.7, 'edge_on_cube': 0.675, 'energy': -1.02}

    #     We need the following again:

    #     {'x0': -0.1, 'x_on_cube0': 0.483, 'sim_id': 0, 'local_min': False,
    #     'local_pt': False, 'x1': 0.7, 'x_on_cube1': 0.675, 'f': -1.02}

    #     """
    #     new_results = []
    #     for entry in results:  # get a dict

    #         new_entry = {}
    #         for map_key in self._vars_x_mapping.keys():  # get 0, 1

    #             for out_key in entry.keys():  # get core, core_on_cube, energy, sim_id, etc.

    #                 # continue over cases where e.g. the map_key may be 0 but we're looking at x1
    #                 if out_key[-1].isnumeric() and not out_key.endswith(str(map_key)):
    #                     continue

    #                 if self._vars_x_mapping[map_key] == out_key:  # found core
    #                     new_name = self._internal_variable + str(map_key)  # create x0, x1, etc.

    #                 # we need to strip trailing ints for this condition in case vars were formatted: x0, x1
    #                 # avoid the "x0_on_cube0" naming scheme
    #                 elif out_key.startswith(self._vars_x_mapping[map_key].rstrip("0123456789")):  # found core_on_cube
    #                     new_name = out_key.replace(
    #                         self._vars_x_mapping[map_key].rstrip("0123456789"), self._internal_variable
    #                     )
    #                     # presumably multi-dim key; preserve that trailing int on the end of new key
    #                     if not new_name[-1].isnumeric():
    #                         new_name += str(map_key)  # create x_on_cube0

    #                 elif out_key in list(self.objectives.keys()):  # found energy
    #                     new_name = self._internal_objective  # create f

    #                 elif out_key in self.gen_specs["persis_in"]:  # found everything else, sim_id, local_pt, etc.
    #                     new_name = out_key

    #                 new_entry[new_name] = entry[out_key]
    #         new_results.append(new_entry)

    #     return new_results

    @abstractmethod
    def ask_numpy(self, num_points: Optional[int] = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""

    @abstractmethod
    def tell_numpy(self, results: npt.NDArray) -> None:
        """Send the results, as a NumPy array, of evaluations to the generator."""

    @staticmethod
    def convert_np_types(dict_list):
        return [
            {key: (value.item() if isinstance(value, np.generic) else value) for key, value in item.items()}
            for item in dict_list
        ]

    def ask(self, num_points: Optional[int] = 0) -> List[dict]:
        """Request the next set of points to evaluate."""
        return LibensembleGenerator.convert_np_types(np_to_list_dicts(self.ask_numpy(num_points)))

    def tell(self, results: List[dict]) -> None:
        """Send the results of evaluations to the generator."""
        self.tell_numpy(list_dicts_to_np(self._objs_and_vars_to_gen_in(results)))


class LibensembleGenThreadInterfacer(LibensembleGenerator):
    """Implement ask/tell for traditionally written libEnsemble persistent generator functions.
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
        self.thread = None

    def setup(self) -> None:
        """Must be called once before calling ask/tell. Initializes the background thread."""
        self.m = Manager()
        self.inbox = self.m.Queue()
        self.outbox = self.m.Queue()

        comm = QComm(self.inbox, self.outbox)
        self.libE_info["comm"] = comm  # replacing comm so gen sends HERE instead of manager
        self.libE_info["executor"] = Executor.executor

        self.thread = QCommProcess(  # TRY A PROCESS
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
            try:
                new_results[field] = results[field]
            except ValueError:  # lets not slot in data that the gen doesnt need?
                continue
        new_results["sim_ended"] = True
        return new_results

    def tell(self, results: List[dict], tag: int = EVAL_GEN_TAG) -> None:
        """Send the results of evaluations to the generator."""
        self.tell_numpy(list_dicts_to_np(self._objs_and_vars_to_gen_in(results)), tag)

    def ask_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if self.thread is None or not self.thread.running:
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
            self.inbox.put((0, np.copy(results)))
        else:
            self.inbox.put((tag, None))

    def final_tell(self, results: npt.NDArray = None) -> (npt.NDArray, dict, int):
        """Send any last results to the generator, and it to close down."""
        self.tell_numpy(results, PERSIS_STOP)  # conversion happens in tell
        return self.thread.result()
