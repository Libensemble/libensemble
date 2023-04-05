import importlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Union

import tomli
import yaml

from libensemble import logger
from libensemble.libE import libE
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

ATTR_ERR_MSG = 'Unable to load "{}". Is the function or submodule correctly named?'
ATTR_ERR_MSG = "\n" + 10 * "*" + ATTR_ERR_MSG + 10 * "*" + "\n"

NOTFOUND_ERR_MSG = 'Unable to load "{}". Is the package installed or the relative path correct?'
NOTFOUND_ERR_MSG = "\n" + 10 * "*" + NOTFOUND_ERR_MSG + 10 * "*" + "\n"


@dataclass
class Ensemble:
    """
    An alternative interface for
    parameterizing libEnsemble by interacting with a class instance, and
    potentially populating it via a yaml, json, or toml file.
    """

    libE_specs: Union[LibeSpecs, dict] = field(default_factory=dict)
    sim_specs: Union[SimSpecs, dict] = field(default_factory=dict)
    gen_specs: Union[GenSpecs, dict] = field(default_factory=dict)
    exit_criteria: Union[ExitCriteria, dict] = field(default_factory=dict)
    persis_info: dict = field(default_factory=dict)
    alloc_specs: Union[AllocSpecs, dict] = field(default_factory=AllocSpecs)
    H0: Any = None

    def __post_init__(self):
        self.nworkers, self.is_manager, libE_specs_parsed, _ = parse_args()
        if isinstance(self.libE_specs, dict) and not len(self.libE_specs):
            self.libE_specs.update(libE_specs_parsed)
        self._util_logger = logging.getLogger(__name__)
        self.logger = logger
        self.logger.set_level("INFO")
        if isinstance(self.libE_specs, dict):
            self.libE_specs.update(LibeSpecs(**self.libE_specs).dict())

        self.corresponding_classes = {
            "sim_specs": SimSpecs,
            "gen_specs": GenSpecs,
            "alloc_specs": AllocSpecs,
            "exit_criteria": ExitCriteria,
            "libE_specs": LibeSpecs,
        }

    def run(self):
        """
        Initializes libEnsemble, passes in all specification dictionaries.
        Sets Ensemble instance's output ``H``, final ``persis_info`` state, and ``flag``.
        Spec checking (and other error handling) occurs within ``libE()``.
        """

        self._cleanup()

        self.H, self.persis_info, self.flag = libE(
            self.sim_specs,
            self.gen_specs,
            self.exit_criteria,
            persis_info=self.persis_info,
            alloc_specs=self.alloc_specs,
            libE_specs=self.libE_specs,
            H0=self.H0,
        )

        return self.H, self.persis_info, self.flag

    def _nworkers(self):
        if self.nworkers:
            return self.nworkers
        elif isinstance(self.libE_specs, dict) and self.libE_specs.get("nworkers"):
            return self.libE_specs["nworkers"]
        elif isinstance(self.libE_specs, LibeSpecs):
            return self.libE_specs.nworkers

    def _get_func(self, loaded):
        """Extracts user function specified in loaded dict"""
        func_path_split = loaded.rsplit(".", 1)
        func_name = func_path_split[-1]
        try:
            return getattr(importlib.import_module(func_path_split[0]), func_name)
        except AttributeError:
            self._util_logger.manager_warning(ATTR_ERR_MSG.format(func_name))
            raise
        except ModuleNotFoundError:
            self._util_logger.manager_warning(NOTFOUND_ERR_MSG.format(func_name))
            raise

    @staticmethod
    def _get_outputs(loaded):
        """Extracts output parameters from loaded dict"""
        if not loaded:
            return []
        fields = [i for i in loaded]
        field_params = [i for i in loaded.values()]
        results = []
        for i in range(len(fields)):
            field_type = field_params[i]["type"]
            built_in_type = __builtins__.get(field_type, field_type)
            try:
                if field_params[i]["size"] == 1:
                    size = (1,)  # formatting how size=1 is typically preferred
                else:
                    size = field_params[i]["size"]
                results.append((fields[i], built_in_type, size))
            except KeyError:
                results.append((fields[i], built_in_type))
        return results

    @staticmethod
    def _get_normal(loaded):
        return loaded

    def _get_option(self, specs, name):
        """Gets a specs value, underlying spec is either a dict or a class"""
        attr = getattr(self, specs)
        if isinstance(attr, dict):
            return attr.get(name)
        else:
            return getattr(attr, name)

    def _parse_spec(self, loaded_spec):
        """Parses and creates traditional libEnsemble dictionary from loaded dict info"""

        field_f = {
            "sim_f": self._get_func,
            "gen_f": self._get_func,
            "alloc_f": self._get_func,
            "inputs": self._get_normal,
            "persis_in": self._get_normal,
            "out": self._get_outputs,
            "funcx_endpoint": self._get_normal,
            "user": self._get_normal,
        }

        userf_fields = [f for f in loaded_spec if f in field_f.keys()]

        if len(userf_fields):
            for f in userf_fields:
                if f == "inputs":
                    loaded_spec["in"] = field_f[f](loaded_spec[f])
                    loaded_spec.pop("inputs")
                else:
                    loaded_spec[f] = field_f[f](loaded_spec[f])

        return loaded_spec

    def _cleanup(self):
        if isinstance(self.libE_specs, dict):
            self.libE_specs.update(LibeSpecs(**self.libE_specs).dict())

        # libE isn't especially instrumented currently to handle "None" exit_criteria values
        if isinstance(self.exit_criteria, dict):
            self.exit_criteria = {k: v for k, v in self.exit_criteria.items() if v is not None}

    def _parameterize(self, loaded):
        """Updates and sets attributes from specs loaded from file"""
        for f in loaded:
            loaded_spec = self._parse_spec(loaded[f])
            old_spec = getattr(self, f)
            ClassType = self.corresponding_classes[f]
            if isinstance(old_spec, dict):
                old_spec.update(loaded_spec)
                if old_spec.get("in") and old_spec.get("inputs"):
                    old_spec.pop("inputs")  # avoid clashes
                setattr(self, f, ClassType(**old_spec).dict())
            else:
                ClassType = self.corresponding_classes[f]
                setattr(self, f, ClassType(**old_spec))

        self._cleanup()

    def from_yaml(self, file_path: str):
        """Parameterizes libEnsemble from yaml file"""
        with open(file_path, "r") as f:
            loaded = yaml.full_load(f)

        self._parameterize(loaded)

    def from_toml(self, file_path: str):
        """Parameterizes libEnsemble from toml file"""
        with open(file_path, "rb") as f:
            loaded = tomli.load(f)

        self._parameterize(loaded)

    def from_json(self, file_path: str):
        """Parameterizes libEnsemble from json file"""
        with open(file_path, "rb") as f:
            loaded = json.load(f)

        self._parameterize(loaded)

    def add_random_streams(self, num_streams: int = 0, seed: str = ""):
        """Adds np.random generators for each worker to persis_info"""
        if num_streams:
            nstreams = num_streams
        else:
            nstreams = self._nworkers()

        self.persis_info = add_unique_random_streams({}, nstreams + 1, seed=seed)
        return self.persis_info

    def save_output(self, file: str):
        """
        Class wrapper for ``save_libE_output``.
        If using a workflow_dir, will place with specified filename in that directory
        """
        if self._get_option("libE_specs", "workflow_dir_path"):
            save_libE_output(
                self.H, self.persis_info, file, self.nworkers, dest_path=self.libE_specs.get("workflow_dir_path")
            )
        else:
            save_libE_output(self.H, self.persis_info, file, self.nworkers)
