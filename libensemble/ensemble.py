import logging
import warnings

import numpy.typing as npt

from libensemble._deprecation import LibEnsembleDeprecationWarning
from libensemble.executors import Executor
from libensemble.libE import libE
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs
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

EXIT_CRITERIA_DEPRECATION = (
    "ExitCriteria as a standalone parameter is deprecated as of libEnsemble 2.0 "
    "and will be removed in 2.1. Pass exit criteria directly to run() instead: "
    "ensemble.run(sim_max=100) or ensemble.run(sim_max=100, wallclock_max=3600). "
    "See https://libensemble.readthedocs.io/... for migration guidance."
)

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
    Parses and validates settings and maintains output.

    .. dropdown:: Example
        :open:

        .. code-block:: python
            :linenos:

            from gest_api.vocs import VOCS

            from libensemble import Ensemble
            from libensemble.gen_classes.sampling import UniformSample
            from libensemble.sim_funcs.simple_sim import norm_eval
            from libensemble.specs import GenSpecs, SimSpecs

            sampling = Ensemble(parse_args=True)

            sampling.sim_specs = SimSpecs(
                sim_f=norm_eval,
                inputs=["x"],
                outputs=[("f", float)],
            )

            vocs = VOCS(
                variables={"x": [-3, 3]},
                objectives={"f": "EXPLORE"},
            )

            generator = UniformSample(vocs=vocs)

            sampling.gen_specs = GenSpecs(
                generator=generator,
                batch_size=50,
            )

            if __name__ == "__main__":
                sampling.run(sim_max=100)
                sampling.save_output(__file__)

    Configure by:

    .. dropdown:: Option 1: Providing parameters on instantiation

        .. code-block:: python
            :linenos:

            from libensemble import Ensemble
            from libensemble.specs import SimSpecs
            from my_simulator import sim_find_energy

            sim_specs = SimSpecs(
                sim_f=sim_find_energy,
                inputs=["x"],
                outputs=[("y", float)],
            )

            experiment = Ensemble(sim_specs=sim_specs)

    .. dropdown:: Option 2: Assigning parameters to an instance

        .. code-block:: python
            :linenos:

            from libensemble import Ensemble
            from libensemble.specs import SimSpecs
            from my_simulator import sim_find_energy

            sim_specs = SimSpecs(
                sim_f=sim_find_energy,
                inputs=["x"],
                outputs=[("y", float)],
            )

            experiment = Ensemble()
            experiment.sim_specs = sim_specs


    Parameters
    ----------

    sim_specs: class:`SimSpecs<libensemble.specs.SimSpecs>`

        Specifications for the simulator function.

    gen_specs: class:`GenSpecs<libensemble.specs.GenSpecs>`, Optional

        Specifications for the generator.

    exit_criteria: class:`ExitCriteria<libensemble.specs.ExitCriteria>`

        Tell libEnsemble when to stop a run.

    libE_specs: class:`LibeSpecs<libensemble.specs.LibeSpecs>`, Optional

        Specifications for libEnsemble.

    alloc_specs: class:`AllocSpecs<libensemble.specs.AllocSpecs>`, Optional

        Specifications for the allocation function.

    persis_info: :obj:`dict`, Optional

        Persistent information to be passed between user function instances
        :doc:`(example)<data_structures/persis_info>`

    executor: :class:`Executor<libensemble.executors.executor.Executor>`, Optional

        libEnsemble Executor instance for use within simulator functions or generators.

    H0: `NumPy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_, Optional

        A libEnsemble history to be prepended to this run's history
        :ref:`(example)<funcguides-history>`.

    parse_args: bool, Optional

        Read ``nworkers``, ``comms``, and other arguments from the command-line. For MPI, calculate ``nworkers``
        and set the ``is_manager`` Boolean attribute on MPI rank 0. See the :meth:`parse_args<tools.parse_args>`
        docs for more information.

    """

    def __init__(
        self,
        sim_specs: SimSpecs = SimSpecs(),
        gen_specs: GenSpecs = GenSpecs(),
        exit_criteria: ExitCriteria | None = None,
        libE_specs: LibeSpecs = LibeSpecs(),
        alloc_specs: AllocSpecs = AllocSpecs(),
        persis_info: dict = {},
        executor: Executor | None = None,
        H0: npt.NDArray | None = None,
        parse_args: bool = False,
    ):
        self.sim_specs = sim_specs
        self.gen_specs = gen_specs
        self._exit_criteria = ExitCriteria()
        if exit_criteria is not None:
            if isinstance(exit_criteria, ExitCriteria):
                warnings.warn(EXIT_CRITERIA_DEPRECATION, LibEnsembleDeprecationWarning, stacklevel=2)
            self._exit_criteria = exit_criteria
        self._libE_specs: LibeSpecs = libE_specs
        self.alloc_specs = alloc_specs
        self.persis_info = persis_info
        self.executor = executor
        self.H0 = H0
        self._util_logger = logging.getLogger(__name__)
        self._nworkers = 0
        self.is_manager = False
        self.parsed = False
        self._known_comms: str = ""
        self._has_run_n_evals = False

        if parse_args:
            self._parse_args()
            self.parsed = True

        if self._libE_specs:
            self._known_comms = getattr(self._libE_specs, "comms", "")

        if self._known_comms == "local":
            self.is_manager = True
            if not self.nworkers:
                raise ValueError("nworkers must be specified if comms is 'local'")

        elif self._known_comms == "mpi" and not parse_args:
            # Set internal _nworkers - not libE_specs (avoid "nworkers will be ignored" warning)
            if self._libE_specs:
                self._nworkers, self.is_manager = mpi_init(getattr(self._libE_specs, "mpi_comm", None))

    def _parse_args(self) -> tuple[int, bool, LibeSpecs]:
        # Set internal _nworkers - not libE_specs (avoid "nworkers will be ignored" warning)
        self._nworkers, self.is_manager, libE_specs_parsed, self.extra_args = parse_args_f()

        if not self._libE_specs:
            self._libE_specs = LibeSpecs(**libE_specs_parsed)
        else:
            self._libE_specs.__dict__.update(**libE_specs_parsed)

        return self.nworkers, self.is_manager, self._libE_specs

    def ready(self) -> tuple[bool, list[str]]:
        """Verify that all necessary data has been provided before calling :meth:`run`.

        Performs a pre-flight check on the ensemble configuration, covering:

        - A simulation callable (``sim_f`` or ``simulator``) is set on ``sim_specs``.
        - At least one exit condition is configured on ``exit_criteria``.
        - Workers are available (``nworkers > 0`` for local/threads/tcp comms,
          or MPI comms is set, which infers workers from the MPI communicator).
        - If both ``gen_specs`` and ``sim_specs`` use the classic field-name interface,
          the generator output field names are a superset of the simulator input field names.

        Returns
        -------
        tuple[bool, list[str]]
            A 2-tuple of ``(is_ready, issues)``.
            ``is_ready`` is ``True`` when all checks pass.
            ``issues`` is a list of human-readable strings describing each problem found;
            it is empty when ``is_ready`` is ``True``.

        Example
        -------
        .. code-block:: python

            ok, issues = sampling.ready()
            if not ok:
                for issue in issues:
                    print(f"  - {issue}")
        """
        issues: list[str] = []

        # --- sim_specs: a callable must be set ---
        sim_callable = getattr(self.sim_specs, "sim_f", None) or getattr(self.sim_specs, "simulator", None)
        if not sim_callable:
            issues.append(
                "sim_specs is missing a callable: set 'sim_f' (a function) or 'simulator' (a gest-api object)."
            )

        # --- exit_criteria: at least one stop condition must be set ---
        ec = self.exit_criteria
        if ec is None or not any(
            getattr(ec, field, None) is not None for field in ("sim_max", "gen_max", "wallclock_max", "stop_val")
        ):
            issues.append(
                "exit_criteria has no stop condition: set at least one of "
                "'sim_max', 'gen_max', 'wallclock_max', or 'stop_val' "
                "either on an ExitCriteria object or directly via "
                "ensemble.run(sim_max=..., gen_max=..., ...)."
            )

        # --- workers: must be determinable ---
        comms = getattr(self._libE_specs, "comms", "mpi")
        if comms in ("local", "threads", "tcp"):
            if not self.nworkers:
                issues.append(
                    f"libE_specs.comms is '{comms}' but 'nworkers' is not set. "
                    "Set 'libE_specs.nworkers' or pass '--nworkers N' on the command line."
                )
        # For 'mpi', worker count is derived from the MPI communicator at runtime; no check needed here.

        # --- cross-spec field consistency (classic interface only) ---
        gen_outputs = [f[0] for f in (getattr(self.gen_specs, "outputs", None) or [])]
        sim_inputs = getattr(self.sim_specs, "inputs", None) or []
        if gen_outputs and sim_inputs:
            missing = [field for field in sim_inputs if field not in gen_outputs]
            if missing:
                issues.append(
                    f"sim_specs.inputs requests field(s) {missing} that are not produced "
                    f"by gen_specs.outputs {gen_outputs}. Check that field names match."
                )

        return not issues, issues

    @property
    def libE_specs(self) -> LibeSpecs:
        return self._libE_specs

    @libE_specs.setter
    def libE_specs(self, new_specs):
        # Respect everything if libE_specs isn't set
        if not hasattr(self, "_libE_specs") or not self._libE_specs:
            self._libE_specs = new_specs
            return

        # Cast new libE_specs temporarily to dict
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

    @property
    def exit_criteria(self) -> ExitCriteria:
        return self._exit_criteria

    @exit_criteria.setter
    def exit_criteria(self, value: ExitCriteria | None):
        if isinstance(value, ExitCriteria):
            warnings.warn(EXIT_CRITERIA_DEPRECATION, LibEnsembleDeprecationWarning, stacklevel=2)
        self._exit_criteria = value or ExitCriteria()

    def reset(self) -> None:
        """Reset the ensemble state to allow a fresh, independent run.

        Clears the accumulated history (``H0``) and ``persis_info`` so that
        the next :meth:`run` call starts from a clean slate — as if no
        previous run had occurred.

        Use this between two calls to :meth:`run` when you want **independent**
        runs rather than the default history-chaining behaviour::

            ens.run(sim_max=10)   # first independent run
            ens.reset()           # clear accumulated history
            ens.run(sim_max=20)   # second independent run, H0 is empty again
        """
        self.H0 = None
        self.persis_info = {}

    def _refresh_executor(self):
        Executor.executor = self.executor or Executor.executor

    def run(
        self,
        sim_max: int | None = None,
        gen_max: int | None = None,
        wallclock_max: float | None = None,
        stop_val: tuple[str, float] | None = None,
    ) -> tuple[npt.NDArray, dict, int]:
        """
        Initializes libEnsemble.

        Parameters
        ----------
        sim_max: int, Optional
            Maximum number of new simulation evaluations for this run.
            Overrides ``exit_criteria.sim_max`` for this call only.
        gen_max: int, Optional
            Maximum number of new generator calls for this run.
            Overrides ``exit_criteria.gen_max`` for this call only.
        wallclock_max: float, Optional
            Wallclock timeout in seconds for this run.
            Overrides ``exit_criteria.wallclock_max`` for this call only.
        stop_val: tuple[str, float], Optional
            Stop criterion ``(field, value)`` for this run.
            Overrides ``exit_criteria.stop_val`` for this call only.

        .. dropdown:: MPI/comms Notes

            Manager--worker intercommunications are parsed from the ``comms`` key of
            :ref:`libE_specs<datastruct-libe-specs>`. An MPI runtime is assumed by default
            if ``-n N`` wasn't specified on the command-line or ``comms="local"`` in ``libE_specs``.

            If a MPI communicator was provided in ``libE_specs``, then each ``.run()`` call
            will initiate on a **duplicate** of that communicator.
            Otherwise, a duplicate of ``COMM_WORLD`` will be used.

        .. dropdown:: Substeps / multi-step usage

            Pass exit-criteria kwargs to run a subset of an ensemble at a time.
            The ensemble history (``H0``) is automatically chained across calls::

                sampling = Ensemble(...)
                sampling.sim_specs = SimSpecs(...)
                sampling.gen_specs = GenSpecs(...)

                # Run in three substeps
                sampling.run(sim_max=30)
                # ... adjust generator hyperparameters ...
                sampling.run(sim_max=30)
                sampling.run(sim_max=40)

            From the second call onward, ``libE_specs.final_gen_send`` and
            ``libE_specs.reuse_output_dir`` are automatically set to ``True``
            to support persistent generators across substep runs.

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
        if self._libE_specs and getattr(self._libE_specs, "comms", "") != self._known_comms:
            raise ValueError(CHANGED_COMMS_WARN)

        assert self._libE_specs is not None

        # Merge kwargs into effective exit criteria for this run
        run_kwargs = {
            k: v
            for k, v in {
                "sim_max": sim_max,
                "gen_max": gen_max,
                "wallclock_max": wallclock_max,
                "stop_val": stop_val,
            }.items()
            if v is not None
        }
        if run_kwargs:
            effective_exit = self._exit_criteria.model_copy(update=run_kwargs)
        else:
            effective_exit = self._exit_criteria

        # Only activate final_gen_send/reuse_output_dir for substep (multi-call) runs,
        # i.e. when a prior run() call has already occurred. A single-call run never
        # needs to chain history back to a persistent generator at shutdown.
        if self._has_run_n_evals:
            self._libE_specs.final_gen_send = True
            self._libE_specs.reuse_output_dir = True

        self.H, self.persis_info, self.flag = libE(
            self.sim_specs,
            self.gen_specs,
            effective_exit,
            persis_info=self.persis_info,
            alloc_specs=self.alloc_specs,
            libE_specs=self._libE_specs,
            H0=self.H0,
        )

        # Chain history for next call
        self.H0 = self.H
        self._has_run_n_evals = True

        return self.H, self.persis_info, self.flag

    @property
    def nworkers(self):
        return self._nworkers or self._libE_specs.nworkers

    @nworkers.setter
    def nworkers(self, value):
        self._nworkers = value
        if self._libE_specs:
            self._libE_specs.nworkers = value

    def save_output(self, basename: str, append_attrs: bool = True):
        """
        Writes out History array and persis_info to files.
        If using a ``workflow_dir_path`` in ``libE_specs``, will place with specified filename in that directory.

        Parameters
        ----------

        Format: ``<basename>_results_History_length=<length>_evals=<Completed evals>_ranks=<nworkers>``

        To have the filename be only the basename, set ``append_attrs=False``

        Format: ``<basename>_results_History_length=<length>_evals=<Completed evals>_ranks=<nworkers>``
        """
        if self.is_manager:
            if getattr(self.libE_specs, "workflow_dir_path", False):
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
