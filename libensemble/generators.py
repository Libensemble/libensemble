from abc import abstractmethod
from typing import List, Optional

import numpy as np
from gest_api import Generator
from gest_api.vocs import VOCS
from numpy import typing as npt

from libensemble.comms.comms import QCommProcess  # , QCommThread
from libensemble.executors import Executor
from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP
from libensemble.tools.tools import add_unique_random_streams
from libensemble.utils.misc import list_dicts_to_np, np_to_list_dicts, unmap_numpy_array


class GeneratorNotStartedException(Exception):
    """Exception raised by a threaded/multiprocessed generator upon being suggested without having been started"""


class LibensembleGenerator(Generator):
    """
    Generator interface that accepts the classic History, persis_info, gen_specs, libE_info parameters after VOCS.

    ``suggest/ingest`` methods communicate lists of dictionaries, like the standard.
    ``suggest_numpy/ingest_numpy`` methods communicate numpy arrays containing the same data.

    .. note::
        Most LibensembleGenerator instances operate on "x" for variables and "f" for objectives internally.
        By default we map "x" to the VOCS variables and "f" to the VOCS objectives, which works for most use cases.
        If a given generator iterates internally over multiple, multi-dimensional variables or objectives,
        then providing a custom ``variables_mapping`` is recommended.

        For instance:
            ``variables_mapping = {"x": ["core", "edge"],
                                   "y": ["mirror-x", "mirror-y"],
                                   "f": ["energy"],
                                   "grad": ["grad_x", "grad_y"]}``.
    """

    def __init__(
        self,
        VOCS: VOCS,
        History: npt.NDArray = [],
        persis_info: dict = {},
        gen_specs: dict = {},
        libE_info: dict = {},
        variables_mapping: dict = {},
        **kwargs,
    ):
        self._validate_vocs(VOCS)
        self.VOCS = VOCS
        self.History = History
        self.gen_specs = gen_specs
        self.libE_info = libE_info

        self.variables_mapping = variables_mapping
        if not self.variables_mapping:
            self.variables_mapping = {}
        # Map variables to x if not already mapped
        if "x" not in self.variables_mapping:
            # SH TODO - is this check needed?
            if len(list(self.VOCS.variables.keys())) > 1 or list(self.VOCS.variables.keys())[0] != "x":
                self.variables_mapping["x"] = self._get_unmapped_keys(self.VOCS.variables, "x")
        # Map objectives to f if not already mapped
        if "f" not in self.variables_mapping:
            if (
                len(list(self.VOCS.objectives.keys())) > 1 or list(self.VOCS.objectives.keys())[0] != "f"
            ):  # e.g. {"f": ["f"]} doesn't need mapping
                self.variables_mapping["f"] = self._get_unmapped_keys(self.VOCS.objectives, "f")
        # Map sim_id to _id
        self.variables_mapping["sim_id"] = ["_id"]

        if len(kwargs) > 0:  # so user can specify gen-specific parameters as kwargs to constructor
            if not self.gen_specs.get("user"):
                self.gen_specs["user"] = {}
            self.gen_specs["user"].update(kwargs)
        if not persis_info.get("rand_stream"):
            self.persis_info = add_unique_random_streams({}, 4, seed=4321)[1]
        else:
            self.persis_info = persis_info

    def _validate_vocs(self, vocs) -> None:
        pass

    def _get_unmapped_keys(self, vocs_dict, default_key):
        """Get keys from vocs_dict that aren't already mapped to other keys in variables_mapping."""
        # Get all variables that aren't already mapped to other keys
        mapped_vars = []
        for mapped_list in self.variables_mapping.values():
            mapped_vars.extend(mapped_list)
        unmapped_vars = [v for v in list(vocs_dict.keys()) if v not in mapped_vars]
        return unmapped_vars

    @abstractmethod
    def suggest_numpy(self, num_points: Optional[int] = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""

    @abstractmethod
    def ingest_numpy(self, results: npt.NDArray) -> None:
        """Send the results, as a NumPy array, of evaluations to the generator."""

    @staticmethod
    def _convert_np_types(dict_list):
        return [
            {key: (value.item() if isinstance(value, np.generic) else value) for key, value in item.items()}
            for item in dict_list
        ]

    def suggest(self, num_points: Optional[int] = 0) -> List[dict]:
        """Request the next set of points to evaluate."""
        return LibensembleGenerator._convert_np_types(
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
        VOCS: VOCS,
        History: npt.NDArray = [],
        persis_info: dict = {},
        gen_specs: dict = {},
        libE_info: dict = {},
        **kwargs,
    ) -> None:
        super().__init__(VOCS, History, persis_info, gen_specs, libE_info, **kwargs)
        self.gen_f = gen_specs["gen_f"]
        self.History = History
        self.libE_info = libE_info
        self._running_gen_f = None
        self.gen_result = None

    def setup(self) -> None:
        """Must be called once before calling suggest/ingest. Initializes the background thread."""
        if self._running_gen_f is not None:
            raise RuntimeError("Generator has already been started.")
        # SH this contains the thread lock -  removing.... wrong comm to pass on anyway.
        if hasattr(Executor.executor, "comm"):
            del Executor.executor.comm
        self.libE_info["executor"] = Executor.executor

        self._running_gen_f = QCommProcess(
            self.gen_f,
            None,
            self.History,
            self.persis_info,
            self.gen_specs,
            self.libE_info,
            user_function=True,
        )

        # This can be set here since the object isnt started until the first suggest
        self.libE_info["comm"] = self._running_gen_f.comm

    def _prep_fields(self, results: npt.NDArray) -> npt.NDArray:
        """Filter out fields that are not in persis_in and add sim_ended to the dtype"""
        filtered_dtype = [
            (name, results.dtype[name]) for name in results.dtype.names if name in self.gen_specs["persis_in"]
        ]

        new_dtype = filtered_dtype + [("sim_ended", bool)]
        new_results = np.zeros(len(results), dtype=new_dtype)

        for field in new_results.dtype.names:
            try:
                new_results[field] = results[field]
            except ValueError:
                continue

        new_results["sim_ended"] = True
        return new_results

    def ingest(self, results: List[dict], tag: int = EVAL_GEN_TAG) -> None:
        """Send the results of evaluations to the generator."""
        self.ingest_numpy(list_dicts_to_np(results, mapping=self.variables_mapping), tag)

    def suggest_numpy(self, num_points: int = 0) -> npt.NDArray:
        """Request the next set of points to evaluate, as a NumPy array."""
        if self._running_gen_f is None:
            self.setup()
            self._running_gen_f.run()
        _, suggest_full = self._running_gen_f.recv()
        return suggest_full["calc_out"]

    def ingest_numpy(self, results: npt.NDArray, tag: int = EVAL_GEN_TAG) -> None:
        """Send the results of evaluations to the generator, as a NumPy array."""
        if self._running_gen_f is None:
            self.setup()
            self._running_gen_f.run()

        if results is not None:
            results = self._prep_fields(results)
            Work = {"libE_info": {"H_rows": np.copy(results["sim_id"]), "persistent": True, "executor": None}}
            self._running_gen_f.send(tag, Work)
            self._running_gen_f.send(tag, np.copy(results))
        else:
            self._running_gen_f.send(tag, None)

    def finalize(self) -> None:
        """Stop the generator process and store the returned data."""
        if self._running_gen_f is None:
            raise RuntimeError("Generator has not been started.")
        self.ingest_numpy(None, PERSIS_STOP)  # conversion happens in ingest
        self.gen_result = self._running_gen_f.result()

    def export(
        self, user_fields: bool = False, as_dicts: bool = False
    ) -> tuple[npt.NDArray | list | None, dict | None, int | None]:
        """Return the generator's results
        Parameters
        ----------
        user_fields : bool, optional
            If True, return local_H with variables unmapped from arrays back to individual fields.
            Default is False.
        as_dicts : bool, optional
            If True, return local_H as list of dictionaries instead of numpy array.
            Default is False.
        Returns
        -------
        local_H : npt.NDArray | list
            Generator history array (unmapped if user_fields=True, as dicts if as_dicts=True).
        persis_info : dict
            Persistent information.
        tag : int
            Status flag (e.g., FINISHED_PERSISTENT_GEN_TAG).
        """
        if not self.gen_result:
            return (None, None, None)
        local_H, persis_info, tag = self.gen_result
        if user_fields and local_H is not None and self.variables_mapping:
            local_H = unmap_numpy_array(local_H, self.variables_mapping)
        if as_dicts and local_H is not None:
            if user_fields and self.variables_mapping:
                local_H = np_to_list_dicts(local_H, self.variables_mapping, allow_arrays=True)
            else:
                local_H = np_to_list_dicts(local_H, allow_arrays=True)
        return (local_H, persis_info, tag)
