import importlib
import json
import logging

import numpy.typing as npt
import tomli
import yaml

from libensemble.executors import Executor
from libensemble.libE import libE
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs
from libensemble.tools import add_unique_random_streams
from libensemble.tools import parse_args as parse_args_f
from libensemble.tools import save_libE_output
from libensemble.tools.parse_args import mpi_init
from libensemble.utils.misc import specs_dump

ATTR_ERR_MSG = 'Unable to load "{}". Is the function or submodule correctly named?'
ATTR_ERR_MSG = "\n" + 10 * "*" + ATTR_ERR_MSG + 10 * "*" + "\n"

NOTFOUND_ERR_MSG = 'Unable to load "{}". Is the package installed or the relative path correct?'
NOTFOUND_ERR_MSG = "\n" + 10 * "*" + NOTFOUND_ERR_MSG + 10 * "*" + "\n"

OVERWRITE_COMMS_WARN = "Cannot reset 'comms' if 'ensemble.libE_specs.comms' is already set."
CHANGED_COMMS_WARN = "New 'comms' method detected following initialization of Ensemble. Exiting."

CORRESPONDING_CLASSES = {
    "sim_specs": SimSpecs,
    "gen_specs": GenSpecs,
    "alloc_specs": AllocSpecs,
    "exit_criteria": ExitCriteria,
    "libE_specs": LibeSpecs,
}


class Ensemble:
    """
    The primary object for a libEnsemble workflow.
    Parses and validates settings, sets up logging, and maintains output.

    .. dropdown:: Example
        :open:

        .. code-block:: python
            :linenos:

            import numpy as np

            from libensemble import Ensemble
            from libensemble.gen_funcs.sampling import latin_hypercube_sample
            from libensemble.sim_funcs.simple_sim import norm_eval
            from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

            libE_specs = LibeSpecs(nworkers=4)
            sampling = Ensemble(libE_specs=libE_specs)
            sampling.sim_specs = SimSpecs(
                sim_f=norm_eval,
                inputs=["x"],
                outputs=[("f", float)],
            )
            sampling.gen_specs = GenSpecs(
                gen_f=latin_hypercube_sample,
                outputs=[("x", float, (1,))],
                user={
                    "gen_batch_size": 50,
                    "lb": np.array([-3]),
                    "ub": np.array([3]),
                },
            )

            sampling.add_random_streams()
            sampling.exit_criteria = ExitCriteria(sim_max=100)

            if __name__ == "__main__":
                sampling.run()
                sampling.save_output(__file__)


    Run the above example via ``python this_file.py``.

    Instead of using the libE_specs line, you can also use ``sampling = Ensemble(parse_args=True)``
    and run via ``python this_file.py -n 4`` (4 workers). The ``parse_args=True`` parameter
    instructs the Ensemble class to read command-line arguments.

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
                outputs=[("y", float)],
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
                        [gen_specs.outputs]
                            [gen_specs.outputs.x]
                                type = "float"
                                size = 1
                        [gen_specs.user]
                            gen_batch_size = 5

                    [sim_specs]
                        sim_f = "simulator.sim_find_sine"
                        inputs = ["x"]
                        [sim_specs.outputs]
                            [sim_specs.outputs.y]
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
                            "outputs": {
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
                            "outputs": {
                                "f": {"type": "float"}
                            }
                        }
                    }

    Parameters
    ----------

    sim_specs: :obj:`dict` or :class:`SimSpecs<libensemble.specs.SimSpecs>`

        Specifications for the simulation function

    gen_specs: :obj:`dict` or :class:`GenSpecs<libensemble.specs.GenSpecs>`, Optional

        Specifications for the generator function

    exit_criteria: :obj:`dict` or :class:`ExitCriteria<libensemble.specs.ExitCriteria>`, Optional

        Tell libEnsemble when to stop a run

    libE_specs: :obj:`dict` or :class:`LibeSpecs<libensemble.specs.LibeSpecs>`, Optional

        Specifications for libEnsemble

    alloc_specs: :obj:`dict` or :class:`AllocSpecs<libensemble.specs.AllocSpecs>`, Optional

        Specifications for the allocation function

    persis_info: :obj:`dict`, Optional

        Persistent information to be passed between user function instances
        :doc:`(example)<data_structures/persis_info>`

    executor: :class:`Executor<libensemble.executors.executor.Executor>`, Optional

        libEnsemble Executor instance for use within simulation or generator functions

    H0: `NumPy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_, Optional

        A libEnsemble history to be prepended to this run's history
        :ref:`(example)<funcguides-history>`

    parse_args: bool, Optional

        Read ``nworkers``, ``comms``, and other arguments from the command-line. For MPI, calculate ``nworkers``
        and set the ``is_manager`` Boolean attribute on MPI rank 0. See the :meth:`parse_args<tools.parse_args>`
        docs for more information.

    """

    def __init__(
        self,
        sim_specs: SimSpecs | None = SimSpecs(),
        gen_specs: GenSpecs | None = GenSpecs(),
        exit_criteria: ExitCriteria | None = {},
        libE_specs: LibeSpecs | None = LibeSpecs(),
        alloc_specs: AllocSpecs | None = AllocSpecs(),
        persis_info: dict | None = {},
        executor: Executor | None = None,
        H0: npt.NDArray | None = None,
        parse_args: bool | None = False,
    ):
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self.exit_criteria = exit_criteria
        self._libE_specs = libE_specs
        self.alloc_specs = alloc_specs
        self.persis_info = persis_info
        self.executor = executor
        self.H0 = H0
        self._util_logger = logging.getLogger(__name__)
        self._nworkers = 0
        self.is_manager = False
        self.parsed = False
        self._known_comms = None

        if parse_args:
            self._parse_args()
            self.parsed = True
            self._known_comms = self._libE_specs.comms

        if not self._known_comms and self._libE_specs is not None:
            if isinstance(self._libE_specs, dict):
                self._libE_specs = LibeSpecs(**self._libE_specs)
            self._known_comms = self._libE_specs.comms

        if self._known_comms == "local":
            self.is_manager = True
            if not self.nworkers:
                raise ValueError("nworkers must be specified if comms is 'local'")

        elif self._known_comms == "mpi" and not parse_args:
            # Set internal _nworkers - not libE_specs (avoid "nworkers will be ignored" warning)
            self._nworkers, self.is_manager = mpi_init(self._libE_specs.mpi_comm)

    def _parse_args(self) -> (int, bool, LibeSpecs):
        # Set internal _nworkers - not libE_specs (avoid "nworkers will be ignored" warning)
        self._nworkers, self.is_manager, libE_specs_parsed, self.extra_args = parse_args_f()

        if not self._libE_specs:
            self._libE_specs = LibeSpecs(**libE_specs_parsed)
        else:
            self._libE_specs.__dict__.update(**libE_specs_parsed)

        return self.nworkers, self.is_manager, self._libE_specs

    def ready(self) -> bool:
        """Quickly verify that all necessary data has been provided"""
        return all([i for i in [self.exit_criteria, self._libE_specs, self.sim_specs]])

    @property
    def libE_specs(self) -> LibeSpecs:
        return self._libE_specs

    @libE_specs.setter
    def libE_specs(self, new_specs):
        # We need to deal with libE_specs being specified as dict or class, and
        #   "not" overwrite the internal libE_specs["comms"].

        # Respect everything if libE_specs isn't set
        if not hasattr(self, "_libE_specs") or not self._libE_specs:
            if isinstance(new_specs, dict):
                self._libE_specs = LibeSpecs(**new_specs)
            else:
                self._libE_specs = new_specs
            return

        # Cast new libE_specs temporarily to dict
        if not isinstance(new_specs, dict):  # exclude_defaults should only be enabled with Pydantic v2
            if new_specs.comms != "mpi" and new_specs.comms != self._libE_specs.comms:  # passing in a non-default comms
                raise ValueError(OVERWRITE_COMMS_WARN)
            platform_specs_set = False
            if new_specs.platform_specs != {}:  # bugginess across Pydantic versions for recursively casting to dict
                platform_specs_set = True
                platform_specs = new_specs.platform_specs
            new_specs = specs_dump(new_specs, exclude_none=True, exclude_defaults=True)
            if platform_specs_set:
                new_specs["platform_specs"] = specs_dump(platform_specs, exclude_none=True)

        # Unset "comms" if we already have a libE_specs that contains that field, that came from parse_args
        if new_specs.get("comms") and hasattr(self._libE_specs, "comms"):
            raise ValueError(OVERWRITE_COMMS_WARN)

        self._libE_specs.__dict__.update(**new_specs)

    def _refresh_executor(self):
        Executor.executor = self.executor or Executor.executor

    def run(self) -> (npt.NDArray, dict, int):
        """
        Initializes libEnsemble.

        .. dropdown:: MPI/comms Notes

            Manager--worker intercommunications are parsed from the ``comms`` key of
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

        self._refresh_executor()

        if self._libE_specs.comms != self._known_comms:
            raise ValueError(CHANGED_COMMS_WARN)

        self.H, self.persis_info, self.flag = libE(
            self.sim_specs,
            self.gen_specs,
            self.exit_criteria,
            persis_info=self.persis_info,
            alloc_specs=self.alloc_specs,
            libE_specs=self._libE_specs,
            H0=self.H0,
        )

        return self.H, self.persis_info, self.flag

    @property
    def nworkers(self):
        return self._nworkers or self._libE_specs.nworkers

    @nworkers.setter
    def nworkers(self, value):
        self._nworkers = value
        if self._libE_specs:
            self._libE_specs.nworkers = value

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
            "outputs": self._get_outputs,
            "globus_compute_endpoint": self._get_normal,
            "user": self._get_normal,
        }

        userf_fields = [f for f in loaded_spec if f in field_f.keys()]

        if len(userf_fields):
            for f in userf_fields:
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
                elif old_spec.get("out") and old_spec.get("outputs"):
                    old_spec.pop("outputs")  # avoid clashes
                setattr(self, f, ClassType(**old_spec))
            else:  # None. attribute not set yet
                setattr(self, f, ClassType(**loaded_spec))

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
        """

        Adds ``np.random`` generators for each worker ID to ``self.persis_info``.

        Parameters
        ----------

        num_streams: int, Optional

            Number of matching worker ID and random stream entries to create. Defaults to
            ``self.nworkers``.

        seed: str, Optional

            Seed for NumPy's RNG.

        """
        if num_streams:
            nstreams = num_streams
        else:
            nstreams = self.nworkers

        self.persis_info = add_unique_random_streams(self.persis_info, nstreams + 1, seed=seed)
        return self.persis_info

    def save_output(self, basename: str, append_attrs: bool = True):
        """
        Writes out History array and persis_info to files.
        If using a workflow_dir, will place with specified filename in that directory.

        Parameters
        ----------

        Format: ``<basename>_results_History_length=<length>_evals=<Completed evals>_ranks=<nworkers>``

        To have the filename be only the basename, set append_attrs=False

        Format: ``<basename>_results_History_length=<length>_evals=<Completed evals>_ranks=<nworkers>``
        """
        if self.is_manager:
            if self._get_option("libE_specs", "workflow_dir_path"):
                save_libE_output(
                    self.H,
                    self.persis_info,
                    basename,
                    self.nworkers,
                    dest_path=self.libE_specs.workflow_dir_path,
                    append_attrs=append_attrs,
                )
            else:
                save_libE_output(self.H, self.persis_info, basename, self.nworkers, append_attrs=append_attrs)
