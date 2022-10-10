import yaml
import pprint
import logging
import inspect
import importlib
from libensemble.libE import libE
from libensemble.alloc_funcs import defaults as alloc_defaults
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.version import __version__
from libensemble import logger

ATTR_ERR_MSG = 'Unable to load "{}". Is the function or submodule correctly named?'
ATTR_ERR_MSG = "\n" + 10 * "*" + ATTR_ERR_MSG + 10 * "*" + "\n"

NOTFOUND_ERR_MSG = 'Unable to load "{}". Is the package installed or the relative path correct?'
NOTFOUND_ERR_MSG = "\n" + 10 * "*" + NOTFOUND_ERR_MSG + 10 * "*" + "\n"


class Persis_Info:
    """
    ``persis_info`` persistent information dictionary management class. An
    instance of this (with random streams) is created on initiation of Ensemble,
    since ``persis_info`` is populated like so for most libEnsemble test-cases anyway.
    """

    def __init__(self, nworkers):
        self.nworkers = nworkers
        self.persis_info = {}

    def add_random_streams(self, num_streams=None, seed=""):
        """
        ``Persis_Info`` wrapper for ``add_unique_random_streams``. Attempt
        to simplify call, since most are identical anyway.
        """
        if num_streams:
            nstreams = num_streams
        else:
            nstreams = self.nworkers + 1

        self.persis_info = add_unique_random_streams({}, nstreams, seed=seed)
        # can access immediately, or ignore return by just using as setter
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
        """Initializes an Ensemble instance. ``parse_args() called on instantiation"""
        self.nworkers, self.is_manager, self.libE_specs, _ = parse_args()
        self.persis_info = Persis_Info(self.nworkers)
        self._util_logger = logging.getLogger(__name__)
        self.logger = logger
        self.logger.set_level("INFO")
        self.sim_specs = {
            "sim_f": None,
            "in": None,
            "persis_in": None,
            "out": None,
            "funcx_endpoint": None,
            "user": None,
            "type": "sim",
        }
        self.gen_specs = {
            "gen_f": None,
            "in": None,
            "persis_in": None,
            "out": None,
            "funcx_endpoint": None,
            "user": None,
            "type": "gen",
        }
        self.alloc_specs = {
            "alloc_f": None,
            "out": None,
            "user": None,
            "type": "alloc",
        }
        self.exit_criteria = {}
        self.H0 = None
        self._filename = inspect.stack()[1].filename

    def __str__(self):
        """
        Returns pretty-printed representation of Ensemble object. Depicts libEnsemble
        version, plus representations of major specification dicts.
        """
        info = f"\nlibEnsemble {__version__}\n" + 79 * "*" + "\n"
        info += "\nCalling Script: " + self._filename.split("/")[-1] + "\n"

        dicts = {
            "libE_specs": self.libE_specs,
            "sim_specs": self.sim_specs,
            "gen_specs": self.gen_specs,
            "alloc_specs": self.alloc_specs,
            "persis_info": self.persis_info.persis_info,
            "exit_criteria": self.exit_criteria,
        }

        for i in dicts:
            info += f"{i}:\n {pprint.pformat(dicts[i])} \n\n"

        info += 79 * "*"
        return info

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

    def _get_func(self, loaded, type):
        """Extracts user function specified in loaded yaml dict"""
        func_path_split = loaded[type + "_specs"]["function"].rsplit(".", 1)
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
    def _get_inputs(loaded, type):
        """Extracts input parameters from loaded yaml dict"""
        return [i for i in loaded[type + "_specs"].get("inputs", [])]

    @staticmethod
    def _get_persis_inputs(loaded, type):
        """Extracts persis input parameters from loaded yaml dict"""
        return [i for i in loaded[type + "_specs"].get("persistent_inputs", [])]

    @staticmethod
    def _get_outputs(loaded, type):
        """Extracts output parameters from loaded yaml dict"""
        outputs = loaded[type + "_specs"].get("outputs")
        if not outputs:
            return []
        fields = [i for i in outputs]
        field_params = [i for i in outputs.values()]
        results = []
        for i in range(len(fields)):
            field_type = field_params[i]["type"]
            # If not a builtin datatype, *probably* a numpy datatype, e.g. "U70" for strings
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
    def _get_endpoint(loaded, type):
        """Extracts funcX endpoint from loaded yaml dict"""
        return loaded[type + "_specs"].get("funcx_endpoint", "")

    @staticmethod
    def _get_user(loaded, type):
        """Extracts user parameters from loaded yaml dict"""
        return loaded[type + "_specs"].get("user", {})

    def from_yaml(self, file):
        """Populates libEnsemble spec dictionaries from yaml file"""
        with open(file, "r") as f:
            loaded = yaml.full_load(f)

        # Functions that extract specific values from the yaml input file
        key_funcs = {
            "sim_f": self._get_func,
            "gen_f": self._get_func,
            "alloc_f": self._get_func,
            "in": self._get_inputs,
            "persis_in": self._get_persis_inputs,
            "out": self._get_outputs,
            "funcx_endpoint": self._get_endpoint,
            "user": self._get_user,
        }

        for spec in [self.sim_specs, self.gen_specs, self.alloc_specs]:
            for key in spec:
                if key == "type":  # should be last key. Nothing more to do
                    spec.pop("type")  # currently not a valid input
                    break
                # Lookup matching extractor, and set value to extractor's output
                try:
                    spec[key] = key_funcs[key](loaded, spec["type"])
                except KeyError as e:  # if no alloc_specs, want defaults
                    if "alloc_specs" in e.args:
                        self.alloc_specs = alloc_defaults.alloc_specs
                        continue
                    else:
                        raise

        # exit_criteria has been included in libE_specs for space (good idea?!)
        self.exit_criteria = loaded["libE_specs"]["exit_criteria"]
        loaded["libE_specs"].pop("exit_criteria")

        self.libE_specs.update(loaded["libE_specs"])

    def save_output(self, file):
        """Class wrapper for save_libE_output"""
        save_libE_output(self.H, self.persis_info, file, self.nworkers)
