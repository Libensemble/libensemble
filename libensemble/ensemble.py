import importlib
import json
import logging
from typing import Optional

import numpy.typing as npt
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

CORRESPONDING_CLASSES = {
    "sim_specs": SimSpecs,
    "gen_specs": GenSpecs,
    "alloc_specs": AllocSpecs,
    "exit_criteria": ExitCriteria,
    "libE_specs": LibeSpecs,
}


class Ensemble:
    """
    The primary class for a libEnsemble workflow.

    .. dropdown:: Example
        :open:

        .. code-block:: python
            :linenos:

            from libensemble import Ensemble, SimSpecs, GenSpecs, ExitCriteria
            from my_simulator import beamline
            from someones_optimizer import optimize

            experiment = Ensemble()
            experiment.sim_specs = SimSpecs(sim_f=beamline, inputs=["x"], out=[("f", float)])
            experiment.gen_specs = GenSpecs(
                gen_f=optimize,
                inputs=["f"],
                out=[("x", float, (1,))],
                user={
                    "lb": np.array([-3]),
                    "ub": np.array([3]),
                },
            )

            experiment.exit_criteria = ExitCriteria(gen_max=101)
            results = experiment.run()

    Parses ``--comms``, ``--nworkers``,
    and other options from the command-line, validates inputs, configures logging,
    and performs other preparations.

    Call ``.run()`` on the class to start the workflow.

    Configure by:

    .. dropdown:: Option 1: Providing parameters on instantiation

        .. code-block:: python
            :linenos:

            from libensemble import Ensemble
            from my_simulator import sim_find_energy

            sim_specs = {
                "sim_f": sim_find_energy,
                "in": ["x"],
                "out": [("y", float)],
            }

            experiment = Ensemble(sim_specs=sim_specs)

    .. dropdown:: Option 2: Assigning parameters to an instance

        .. code-block:: python
            :linenos:

            from libensemble import Ensemble, SimSpecs
            from my_simulator import sim_find_energy

            sim_specs = SimSpecs(
                sim_f=sim_find_energy,
                inputs=["x"],
                out=[("y", float)],
            )

            experiment = Ensemble()
            experiment.sim_specs = sim_specs

    .. dropdown:: Option 3: Loading parameters from files

        .. code-block:: python
            :linenos:

            from libensemble import Ensemble

            experiment = Ensemble()

            my_experiment.from_yaml("my_parameters.yaml")
            # or...
            my_experiment.from_toml("my_parameters.toml")
            # or...
            my_experiment.from_json("my_parameters.json")

        .. tab-set::

            .. tab-item:: my_parameters.yaml

                .. code-block:: yaml
                    :linenos:

                    libE_specs:
                        save_every_k_gens: 20

                    exit_criteria:
                        sim_max: 80

                    gen_specs:
                        gen_f: generator.gen_random_sample
                        outputs:
                            x:
                                type: float
                                size: 1
                        user:
                            gen_batch_size: 5

                    sim_specs:
                        sim_f: simulator.sim_find_sine
                        inputs:
                            - x
                        outputs:
                            y:
                                type: float

            .. tab-item:: my_parameters.toml

                .. code-block:: toml
                    :linenos:

                    [libE_specs]
                        save_every_k_gens = 300

                    [exit_criteria]
                        sim_max = 80

                    [gen_specs]
                        gen_f = "generator.gen_random_sample"
                        [gen_specs.out]
                            [gen_specs.out.x]
                                type = "float"
                                size = 1
                        [gen_specs.user]
                            gen_batch_size = 5

                    [sim_specs]
                        sim_f = "simulator.sim_find_sine"
                        inputs = ["x"]
                        [sim_specs.out]
                            [sim_specs.out.y]
                                type = "float"

            .. tab-item:: my_parameters.json

                .. code-block:: json
                    :linenos:

                    {
                        "libE_specs": {
                            "save_every_k_gens": 300,
                        },
                        "exit_criteria": {
                            "sim_max": 80
                        },
                        "gen_specs": {
                            "gen_f": "generator.gen_random_sample",
                            "out": {
                                "x": {
                                    "type": "float",
                                    "size": 1
                                }
                            },
                            "user": {
                                "gen_batch_size": 5
                            }
                        },
                        "sim_specs": {
                            "sim_f": "simulator.sim_find_sine",
                            "inputs": ["x"],
                            "out": {
                                "f": {"type": "float"}
                            }
                        }
                    }

    After calling ``.run()``, the final states of ``H``, ``persis_info``,
    and a flag are made available.

    Parameters
    ----------

    sim_specs: :obj:`dict` or :class:`SimSpecs<libensemble.specs.SimSpecs>`

        Specifications for the simulation function

    gen_specs: :obj:`dict` or :class:`GenSpecs<libensemble.specs.GenSpecs>`, optional

        Specifications for the generator function

    exit_criteria: :obj:`dict` or :class:`ExitCriteria<libensemble.specs.ExitCriteria>`, optional

        Tell libEnsemble when to stop a run

    persis_info: :obj:`dict`, optional

        Persistent information to be passed between user functions
        :doc:`(example)<data_structures/persis_info>`

    alloc_specs: :obj:`dict` or :class:`AllocSpecs<libensemble.specs.AllocSpecs>`, optional

        Specifications for the allocation function

    libE_specs: :obj:`dict` or :class:`LibeSpecs<libensemble.specs.libeSpecs>`, optional

        Specifications for libEnsemble

    H0: `NumPy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_, optional

        A libEnsemble history to be prepended to this run's history
        :ref:`(example)<funcguides-history>`

    """

    def __init__(
        self,
        sim_specs: Optional[SimSpecs] = SimSpecs(),
        gen_specs: Optional[GenSpecs] = GenSpecs(),
        exit_criteria: Optional[ExitCriteria] = {},
        libE_specs: Optional[LibeSpecs] = None,
        alloc_specs: Optional[AllocSpecs] = AllocSpecs(),
        persis_info: Optional[dict] = {},
        H0: Optional[npt.NDArray] = None,
    ):
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.exit_criteria = exit_criteria
        self.libE_specs = libE_specs
        self.alloc_specs = alloc_specs
        self.persis_info = persis_info
        self.H0 = H0

        self.nworkers, self.is_manager, libE_specs_parsed, _ = parse_args()
        self._util_logger = logging.getLogger(__name__)
        self.logger = logger
        self.logger.set_level("INFO")

        if not self.libE_specs:
            self.libE_specs = LibeSpecs(**libE_specs_parsed)

    def ready(self) -> bool:
        """Quickly verify that all necessary data has been provided"""
        return all([i for i in [self.exit_criteria, self.libE_specs, self.sim_specs]])

    def run(self) -> (npt.NDArray, dict, int):
        """
        Initializes libEnsemble.

        .. dropdown:: MPI/comms Notes

            Manager-worker intercommunications are parsed from the ``comms`` key of
            :ref:`libE_specs<datastruct-libe-specs>`. An MPI runtime is assumed by default
            if ``--comms local`` wasn't specified on the command-line or in ``libE_specs``.

            If a MPI communicator was provided in ``libE_specs``, then each ``.run()`` call
            will initiate intercommunications on a **duplicate** of that communicator.
            Otherwise, a duplicate of ``COMM_WORLD`` will be used.

        Returns
        -------

        H: NumPy structured array

            History array storing rows for each point.
            :ref:`(example)<funcguides-history>`

        persis_info: :obj:`dict`

            Final state of persistent information
            :doc:`(example)<data_structures/persis_info>`

        exit_flag: :obj:`int`

            Flag containing final task status

            .. code-block::

                0 = No errors
                1 = Exception occurred
                2 = Manager timed out and ended simulation
                3 = Current process is not in libEnsemble MPI communicator
        """

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

    def _parameterize(self, loaded):
        """Updates and sets attributes from specs loaded from file"""
        for f in loaded:
            loaded_spec = self._parse_spec(loaded[f])
            old_spec = getattr(self, f)
            ClassType = CORRESPONDING_CLASSES[f]
            if isinstance(old_spec, dict):
                old_spec.update(loaded_spec)
                if old_spec.get("in") and old_spec.get("inputs"):
                    old_spec.pop("inputs")  # avoid clashes
            else:
                old_spec.__dict__.update(**loaded_spec)
                old_spec = old_spec.dict(by_alias=True)
            setattr(self, f, ClassType(**old_spec))

    def from_yaml(self, file_path: str):
        """Parameterizes libEnsemble from ``yaml`` file"""
        with open(file_path, "r") as f:
            loaded = yaml.full_load(f)

        self._parameterize(loaded)

    def from_toml(self, file_path: str):
        """Parameterizes libEnsemble from ``toml`` file"""
        with open(file_path, "rb") as f:
            loaded = tomli.load(f)

        self._parameterize(loaded)

    def from_json(self, file_path: str):
        """Parameterizes libEnsemble from ``json`` file"""
        with open(file_path, "rb") as f:
            loaded = json.load(f)

        self._parameterize(loaded)

    def add_random_streams(self, num_streams: int = 0, seed: str = ""):
        """Adds ``np.random`` generators for each worker to ``persis_info``"""
        if num_streams:
            nstreams = num_streams
        else:
            nstreams = self._nworkers()

        self.persis_info = add_unique_random_streams({}, nstreams + 1, seed=seed)
        return self.persis_info

    def save_output(self, file: str):
        """
        Writes out History array and persis_info to files.
        If using a workflow_dir, will place with specified filename in that directory

        Format: ``<calling_script>_results_History_length=<length>_evals=<Completed evals>_ranks=<nworkers>``
        """
        if self._get_option("libE_specs", "workflow_dir_path"):
            save_libE_output(self.H, self.persis_info, file, self.nworkers, dest_path=self.libE_specs.workflow_dir_path)
        else:
            save_libE_output(self.H, self.persis_info, file, self.nworkers)
