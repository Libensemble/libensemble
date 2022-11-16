import sys
import yaml
import tomli
import logging
import importlib
from dataclasses import dataclass
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.specs import SimSpecs, GenSpecs, AllocSpecs, LibeSpecs, ExitCriteria, EnsembleSpecs
from libensemble.version import __version__
from libensemble import logger

ATTR_ERR_MSG = 'Unable to load "{}". Is the function or submodule correctly named?'
ATTR_ERR_MSG = "\n" + 10 * "*" + ATTR_ERR_MSG + 10 * "*" + "\n"

NOTFOUND_ERR_MSG = 'Unable to load "{}". Is the package installed or the relative path correct?'
NOTFOUND_ERR_MSG = "\n" + 10 * "*" + NOTFOUND_ERR_MSG + 10 * "*" + "\n"

@dataclass
class Persis_Info:
    """
    ``persis_info`` persistent information dictionary management class. An
    instance of this (with random streams) is created on initiation of Ensemble,
    since ``persis_info`` is populated like so for most libEnsemble test-cases anyway.
    """
    nworkers: int = 4
    persis_info = {}

    def add_random_streams(self, num_streams: int=0, seed=""):
        if num_streams:
            nstreams = num_streams
        else:
            nstreams = self.nworkers + 1

        self.persis_info = add_unique_random_streams({}, nstreams, seed=seed)
        return self.persis_info

class Ensemble:
    """
    The vast majority of libEnsemble cases require the user to instantiate
    and populate a set of specification dictionaries inside a calling script,
    call ``parse_args()``, then call ``libE()`` while passing in each spec
    dictionary. Many calling scripts and ``libE()`` calls are often identical,
    even across widely varying use-cases. This is an alternative interface for
    parameterizing libEnsemble by interacting with a class instance, and
    potentially populating it via a yaml file.
    """

    def __init__(self):
        """Initializes an Ensemble instance. ``parse_args()`` called on instantiation"""
        self.nworkers, self.is_manager, libE_specs_parsed, _ = parse_args()
        self.persis_info = Persis_Info(self.nworkers)
        self._util_logger = logging.getLogger(__name__)
        self.logger = logger
        self.logger.set_level("INFO")
        self.sim_specs: SimSpecs = SimSpecs()
        self.gen_specs: GenSpecs = GenSpecs()
        self.alloc_specs: AllocSpecs = AllocSpecs()
        self.libE_specs: LibeSpecs = LibeSpecs.parse_obj(libE_specs_parsed)
        self.exit_criteria: ExitCriteria = ExitCriteria()
        self.H0 = None

    def run(self):
        """
        Initializes libEnsemble, passes in all specification dictionaries.
        Sets Ensemble instance's output H, final persis_info state, and flag.
        Spec checking (and other error handling) occurs within ``libE()``.
        """

        self.H, self.persis_info.persis_info, self.flag = libE(
            self.sim_specs,
            self.gen_specs,
            self.exit_criteria,
            persis_info=self.persis_info.persis_info,
            alloc_specs=self.alloc_specs,
            libE_specs=self.libE_specs,
            H0=self.H0,
            )

        return self.H, self.persis_info.persis_info, self.flag

    def _get_func(self, loaded):
        """Extracts user function specified in loaded yaml dict"""
        if isinstance(loaded, str):
            func_path_split = loaded.rsplit(".", 1)
        else:
            return loaded
        try:
            func_name = func_path_split[-1]
            return getattr(importlib.import_module(func_path_split[0]), func_name)
        except AttributeError:
            self._util_logger.manager_warning(ATTR_ERR_MSG.format(func_name))
            raise
        except ModuleNotFoundError:
            self._util_logger.manager_warning(NOTFOUND_ERR_MSG.format(func_name))
            raise

    @staticmethod
    def _get_outputs(loaded):
        """Extracts output parameters from loaded yaml dict"""
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

    def _load_spec(self, loaded_spec):
        """Parses and creates traditional libEnsemble dictionary from yaml section"""

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

        userf_fields = [field for field in loaded_spec if field in field_f.keys()]

        if len(userf_fields):
            for field in userf_fields:
                if field == "inputs":
                    loaded_spec["in"] = field_f[field](loaded_spec[field])
                    loaded_spec.pop("inputs")
                else:
                    loaded_spec[field] = field_f[field](loaded_spec[field])

        return loaded_spec

    def from_yaml(self, file):
        """Parameterizes libEnsemble from yaml file"""

        with open(file, "r") as f:
            loaded = yaml.full_load(f)

        specs = {
            "sim_specs": SimSpecs,
            "gen_specs": GenSpecs,
            "alloc_specs": AllocSpecs,
            "libE_specs": LibeSpecs,
            "exit_criteria": ExitCriteria,
        }

        specs_out = {
            "sim_specs": None,
            "gen_specs": None,
            "alloc_specs": None,
            "libE_specs": None,
            "exit_criteria": None,
        }

        # TODO: Replacing classes with instantiated objects???
        for spec in loaded:
            specs_out[spec] = specs[spec].parse_obj(self._load_spec(loaded[spec]))

        # Merge data, need past as dicts first
        sim_specs_p = self.sim_specs.dict(by_alias=True)
        gen_specs_p = self.gen_specs.dict(by_alias=True)
        exit_criteria_p = self.exit_criteria.dict(by_alias=True, exclude_unset=True)
        alloc_specs_p = self.alloc_specs.dict(by_alias=True)
        libE_specs_p = self.libE_specs.dict(by_alias=True)

        # then newly updated as dicts
        sim_specs_f = specs["sim_specs"].dict(by_alias=True)
        gen_specs_f = specs["gen_specs"].dict(by_alias=True)
        exit_criteria_f = specs["exit_criteria"].dict(by_alias=True, exclude_unset=True)
        alloc_specs_f = specs["alloc_specs"].dict(by_alias=True)
        libE_specs_f = specs["libE_specs"].dict(by_alias=True)

        # then update on each
        sim_specs_p.update(sim_specs_f)
        gen_specs_p.update(gen_specs_f)
        exit_criteria_p.update(exit_criteria_f)
        alloc_specs_p.update(alloc_specs_f)
        libE_specs_p.update(libE_specs_f)

        # then set back as classes
        self.sim_specs = SimSpecs.parse_obj(sim_specs_p)
        self.gen_specs = GenSpecs.parse_obj(gen_specs_p)
        self.alloc_specs = AllocSpecs.parse_obj(exit_criteria_p)
        self.libE_specs = LibeSpecs.parse_obj(alloc_specs_p)
        self.exit_criteria = ExitCriteria.parse_obj(libE_specs_p)

    def save_output(self, file):
        """Class wrapper for save_libE_output"""
        save_libE_output(self.H, self.persis_info, file, self.nworkers)
