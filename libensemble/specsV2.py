import random
import secrets
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.gen_funcs.sampling import latin_hypercube_sample
from libensemble.resources.platformsV2 import Platform
from libensemble.sim_funcs.one_d_func import one_d_example
from libensemble.utils.specs_checkers import (
    _check_any_workers_and_disable_rm_if_tcp,
    _check_exit_criteria,
    _check_H0,
    _check_output_fields,
)

_OUT_DTYPE_ERR = "unable to coerce into a NumPy dtype. It should be a list of 2-tuples or 3-tuples"
_IN_INVALID_ERR = "value should be a list of field names (a list of strings)"

__all__ = ["SimSpecs", "GenSpecs", "AllocSpecs", "ExitCriteria", "LibeSpecs", "_EnsembleSpecs"]


class SimSpecs(BaseModel):
    """
    Specifications for configuring a Simulation Function.
    """

    sim_f: Callable = one_d_example
    """
    Python function matching the ``sim_f`` interface. Evaluates parameters
    produced by a generator function.
    """

    inputs: List[str] = Field([], alias="in")
    """
    List of **field names** out of the complete history to pass
    into the simulation function upon calling.
    """

    persis_in: Optional[List[str]] = []
    """
    List of **field names** to send to a persistent simulation function
    throughout the run, following initialization.
    """

    # list of tuples for dtype construction
    outputs: List = Field([], alias="out")
    """
    List of 2- or 3-tuples corresponding to NumPy dtypes.
    e.g. ``("dim", int, (3,))``, or ``("path", str)``.
    Typically used to initialize an output array within the simulation function:
    ``out = np.zeros(100, dtype=sim_specs["out"])``.
    Also necessary to construct libEnsemble's history array.
    """

    globus_compute_endpoint: Optional[str] = ""
    """
    A Globus Compute (https://www.globus.org/compute) ID corresponding to an active endpoint on a remote system.
    libEnsemble's workers will submit simulator function instances to this endpoint instead of
    calling them locally.
    """

    user: Optional[dict] = {}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters for customizing
    the simulator function.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    @field_validator("outputs")
    @classmethod
    def check_valid_out(cls, v):
        try:
            _ = np.dtype(v)
        except TypeError:
            raise ValueError(_OUT_DTYPE_ERR)
        else:
            return v

    @field_validator("inputs", "persis_in", mode="before")
    @classmethod
    def check_valid_in(cls, v):
        if not all(isinstance(s, str) for s in v):
            raise ValueError(_IN_INVALID_ERR)
        return v


class GenSpecs(BaseModel):
    """
    Specifications for configuring a Generator Function.
    """

    gen_f: Callable = latin_hypercube_sample
    """
    Python function matching the ``gen_f`` interface. Produces parameters for evaluation by a
    simulator function, and makes decisions based on simulator function output.
    """

    inputs: Optional[List[str]] = Field([], alias="in")
    """
    List of **field names** out of the complete history to pass
    into the generator function upon calling.
    """

    persis_in: Optional[List[str]] = []
    """
    List of **field names** to send to a persistent generator function
    throughout the run, following initialization.
    """

    outputs: List = Field([], alias="out")
    """
    List of 2- or 3-tuples corresponding to NumPy dtypes.
    e.g. ``("dim", int, (3,))``, or ``("path", str)``. Typically used to initialize an
    output array within the generator: ``out = np.zeros(100, dtype=gen_specs["out"])``.
    Also used to construct libEnsemble's history array.
    """

    globus_compute_endpoint: Optional[str] = ""
    """
    A Globus Compute (https://www.globus.org/compute) ID corresponding to an active endpoint on a remote system.
    libEnsemble's workers will submit generator function instances to this endpoint instead of
    calling them locally.
    """

    user: Optional[dict] = {}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters for
    customizing the generator function
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    @field_validator("outputs", mode="before")
    @classmethod
    def check_valid_out(cls, v):
        try:
            _ = np.dtype(v)
        except TypeError:
            raise ValueError(_OUT_DTYPE_ERR)
        else:
            return v

    @field_validator("inputs", "persis_in", mode="before")
    @classmethod
    def check_valid_in(cls, v):
        if not all(isinstance(s, str) for s in v):
            raise ValueError(_IN_INVALID_ERR)
        return v


class AllocSpecs(BaseModel):
    """
    Specifications for configuring an Allocation Function.
    """

    alloc_f: Callable = give_sim_work_first
    """
    Python function matching the ``alloc_f`` interface. Decides when simulator and generator functions
    should be called, and with what resources and parameters.
    """

    user: Optional[dict] = {"num_active_gens": 1}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters
    for customizing the allocation function.
    """

    outputs: Optional[List] = Field([], alias="out")
    """
    List of 2- or 3-tuples corresponding to NumPy dtypes. e.g. ``("dim", int, (3,))``, or ``("path", str)``.
    Allocation functions that modify libEnsemble's History array with additional fields should list those
    fields here. Also used to construct libEnsemble's history array.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )
    # end_alloc_tag


class ExitCriteria(BaseModel):
    """
    Specifications for configuring when libEnsemble should stop a given run.
    """

    sim_max: Optional[int] = None
    """Stop when this many new points have been evaluated by simulation functions."""

    gen_max: Optional[int] = None
    """Stop when this many new points have been generated by generator functions."""

    wallclock_max: Optional[float] = None
    """Stop when this many seconds has elapsed since the manager initialized."""

    stop_val: Optional[Tuple[str, float]] = None
    """Stop when ``H[str] < float`` for the given ``(str, float)`` pair."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )


class LibeSpecs(BaseModel):
    """
    Specifications for configuring libEnsemble's runtime behavior.
    """

    comms: Optional[str] = "mpi"
    """ Manager/Worker communications mode. ``'mpi'``, ``'local'``, or ``'tcp'`` """

    nworkers: Optional[int] = None
    """ Number of worker processes in ``"local"`` or ``"tcp"``."""

    mpi_comm: Optional[Any] = None
    """ libEnsemble MPI communicator. Default: ``MPI.COMM_WORLD``"""

    dry_run: Optional[bool] = False
    """ Whether libEnsemble should immediately exit after validating all inputs. """

    abort_on_exception: Optional[bool] = True
    """
    In MPI mode, whether to call ``MPI_ABORT`` on an exception.
    If ``False``, an exception will be raised by the manager.
    """

    save_every_k_sims: Optional[int] = 0
    """ Save history array to file after every k evaluated points. """

    save_every_k_gens: Optional[int] = 0
    """ Save history array to file after every k generated points. """

    save_H_and_persis_on_abort: Optional[bool] = True
    """ Save states of ``H`` and ``persis_info`` to file on aborting after an exception."""

    save_H_on_completion: Optional[bool] = False
    """
    Save state of ``H`` to file upon completing a workflow. Also enabled when either ``save_every_k_sims``
    or ``save_every_k_gens`` is set.
    """

    save_H_with_date: Optional[bool] = False
    """ ``H`` filename contains date and timestamp."""

    H_file_prefix: Optional[str] = "libE_history"
    """ Prefix for ``H`` filename."""

    worker_timeout: Optional[int] = 1
    """ On libEnsemble shutdown, number of seconds after which workers considered timed out, then terminated. """

    kill_canceled_sims: Optional[bool] = False
    """
    Try to kill sims with ``"cancel_requested"`` set ``True``.
    If ``False``, the manager avoids this moderate overhead.
    """

    use_workflow_dir: Optional[bool] = False
    """
    Whether to place *all* log files, dumped arrays, and default output directories in a
    separate `workflow` directory. Each run will be suffixed with a hash.
    If copying back an ensemble directory from a scratch space, the copy is placed here.
    """

    reuse_output_dir: Optional[bool] = False
    """
    Whether to allow overwrites and access to previous ensemble and workflow directories in subsequent runs.
    ``False`` by default to protect results.
    """

    workflow_dir_path: Optional[Union[str, Path]] = "."
    """
    Optional path to the workflow directory.
    """

    ensemble_dir_path: Optional[Union[str, Path]] = Path("ensemble")
    """
    Path to main ensemble directory. Can serve
    as a single working directory for workers, or contain calculation directories
    """

    ensemble_copy_back: Optional[bool] = False
    """
    Whether to copy back contents of ``ensemble_dir_path`` to launch
    location. Useful if ``ensemble_dir_path`` is located on node-local storage.
    """

    use_worker_dirs: Optional[bool] = False
    """ Whether to organize calculation directories under worker-specific directories. """

    sim_dirs_make: Optional[bool] = False
    """
    Whether to make calculation directories for each simulation function call.
    """

    sim_dir_copy_files: Optional[List[Union[str, Path]]] = []
    """
    Paths to copy into the working directory upon calling the simulation function.
    List of strings or ``pathlib.Path`` objects.
    """

    sim_dir_symlink_files: Optional[List[Union[str, Path]]] = []
    """
    Paths to symlink into the working directory upon calling the simulation function.
    List of strings or ``pathlib.Path`` objects.
    """

    sim_input_dir: Optional[Union[str, Path]] = None
    """
    Copy this directory's contents into the working directory upon calling the simulation function.
    """

    gen_dirs_make: Optional[bool] = False
    """
    Whether to make generator-specific calculation directories for each generator function call.
    """

    gen_dir_copy_files: Optional[List[Union[str, Path]]] = []
    """
    Paths to copy into the working directory upon calling the generator function.
    List of strings or ``pathlib.Path`` objects
    """

    gen_dir_symlink_files: Optional[List[Union[str, Path]]] = []
    """
    Paths to symlink into the working directory upon calling the generator function.
    List of strings or ``pathlib.Path`` objects.
    """

    gen_input_dir: Optional[Union[str, Path]] = None
    """
    Copy this directory's contents into the working directory upon calling the generator function.
    """

    calc_dir_id_width: Optional[int] = 4
    """
    The width of the numerical ID component of a calculation directory name. Leading
    zeros are padded to the sim/gen ID.
    """

    platform: Optional[str] = ""
    """Name of a known platform defined in the platforms module.

    See :class:`Known Platforms List<libensemble.resources.platforms.Known_platforms>`.

    Example:

    .. code-block:: python

        libE_specs["platform"] = "perlmutter_g"

    Alternatively set the environment variable ``LIBE_PLATFORM``:

    .. code-block:: shell

        export LIBE_PLATFORM="perlmutter_g"

    See also option :attr:`platform_specs`.
    """

    platform_specs: Optional[Union[Platform, dict]] = {}
    """A Platform object or dictionary specifying settings for a platform.

    To use existing platform:

    .. code-block:: python

        from libensemble.resources.platforms import PerlmutterGPU

        libE_specs["platform_specs"] = PerlmutterGPU()

    See :class:`Known Platforms List<libensemble.resources.platforms.Known_platforms>`.

    Or define a platform:

    .. code-block:: python

        from libensemble.resources.platforms import Platform

        libE_specs["platform_specs"] = Platform(
            mpi_runner="srun",
            cores_per_node=64,
            logical_cores_per_node=128,
            gpus_per_node=8,
            gpu_setting_type="runner_default",
            scheduler_match_slots=False,
        )

    For list of Platform fields see :class:`Platform Fields<libensemble.resources.platforms.Platform>`.

    Any fields not given will be auto-detected by libEnsemble.

    See also option :attr:`platform`.
    """

    profile: Optional[bool] = False
    """ Profile manager and worker logic using ``cProfile``. """

    disable_log_files: Optional[bool] = False
    """ Disable ``ensemble.log`` and ``libE_stats.txt`` log files. """

    safe_mode: Optional[bool] = False
    """ Prevents user functions from overwriting protected History fields, but requires moderate overhead. """

    stats_fmt: Optional[dict] = {}
    """ Options for formatting ``'libE_stats.txt'``. See 'Formatting libE_stats.txt'. """

    workers: Optional[List[str]] = None
    """ TCP Only: A list of worker hostnames. """

    ip: Optional[str] = None
    """ TCP Only: IP address for Manager's system. """

    port: Optional[int] = 0
    """ TCP Only: Port number for Manager's system. """

    authkey: Optional[str] = f"libE_auth_{random.randrange(99999)}"
    """ TCP Only: Authkey for Manager's system."""

    workerID: Optional[int] = None
    """ TCP Only: Worker ID number assigned to the new process. """

    worker_cmd: Optional[List[str]] = None
    """
    TCP Only: Split string corresponding to worker/client Python process invocation. Contains
    a local Python path, calling script, and manager/server format-fields for ``manager_ip``,
    ``manager_port``, ``authkey``, and ``workerID``. ``nworkers`` is specified normally.
    """

    use_persis_return_gen: Optional[bool] = False
    """ Adds persistent generator output fields to the History array on return. """

    use_persis_return_sim: Optional[bool] = False
    """ Adds persistent simulator output fields to the History array on return. """

    final_gen_send: Optional[bool] = False
    """
    Send final simulation results to persistent generators before shutdown.
    The results will be sent along with the ``PERSIS_STOP`` tag.
    """

    disable_resource_manager: Optional[bool] = False
    """
    Disable the built-in resource manager, including automatic resource detection
    and/or assignment of resources to workers. ``"resource_info"`` will be ignored.
    """

    num_resource_sets: Optional[int] = None
    """
    Total number of resource sets. Resources will be divided into this number.
    If not set, resources will be divided evenly (excluding zero_resource_workers).
    """

    gen_num_procs: Optional[int] = None
    """
    The default number of processors (MPI ranks) required by generators. Unless
    overridden by the equivalent `persis_info` settings, generators will be
    allocated this many processors for applications launched via the MPIExecutor.
    """

    gen_num_gpus: Optional[int] = None
    """
    The default number of GPUs required by generators. Unless overridden by
    the equivalent `persis_info` settings, generators will be allocated this
    many GPUs.
    """

    enforce_worker_core_bounds: Optional[bool] = False
    """
    If ``False``, the Executor will permit the submission of tasks with a
    higher processor count than the CPUs available to the worker as
    detected by the resource manager. Larger node counts are not allowed.
    When ``"disable_resource_manager"`` is ``True``,
    this argument is ignored
    """

    dedicated_mode: Optional[bool] = False
    """
    Instructs libEnsemble to not run applications on resources where libEnsemble
    processes (manager and workers) are running.
    """

    zero_resource_workers: Optional[List[int]] = []
    """
    List of workers that require no resources. For when a fixed mapping of workers
    to resources is required. Otherwise, use ``num_resource_sets``.
    For use with supported allocation functions.
    """

    resource_info: Optional[dict] = {}
    """
    Resource information to override automatically detected resources.
    Allowed fields are given below in 'Overriding Resource Auto-detection'.
    Note that if ``disable_resource_manager`` is set then this option is ignored.
    """

    scheduler_opts: Optional[dict] = {}
    """ Options for the resource scheduler. See 'Scheduler Options' for more info """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    @field_validator("comms")
    @classmethod
    def check_valid_comms_type(cls, value):
        assert value in ["mpi", "local", "tcp"], "Invalid comms type"
        return value

    @field_validator("platform_specs")
    @classmethod
    def set_platform_specs_to_class(cls, value: Union[Platform, dict]) -> Platform:
        if isinstance(value, dict):
            value = Platform(**value)
        return value

    @field_validator("sim_input_dir", "gen_input_dir")
    @classmethod
    def check_input_dir_exists(cls, value):
        if value:
            if isinstance(value, str):
                value = Path(value).absolute()
            assert value.exists(), "value does not refer to an existing path"
            assert value != Path("."), "Value can't refer to the current directory ('.' or Path('.'))."
        return value

    @field_validator("sim_dir_copy_files", "sim_dir_symlink_files", "gen_dir_copy_files", "gen_dir_symlink_files")
    @classmethod
    def check_inputs_exist(cls, value):
        value = [Path(path).absolute() for path in value]
        for f in value:
            assert f.exists(), f"'{f}' in Value does not refer to an existing path."
        return value

    @model_validator(mode="after")
    def check_any_workers_and_disable_rm_if_tcp(self):
        return _check_any_workers_and_disable_rm_if_tcp(self)

    @model_validator(mode="after")
    def enable_save_H_when_every_K(self):
        if "save_H_on_completion" not in self.__dict__ and (
            self.__dict__.get("save_every_k_sims", 0) > 0 or self.__dict__.get("save_every_k_gens", 0) > 0
        ):
            self.__dict__["save_H_on_completion"] = True
            return self

    @model_validator(mode="after")
    def set_workflow_dir(self):
        if self.use_workflow_dir and len(str(self.workflow_dir_path)) <= 1:
            self.__dict__["workflow_dir_path"] = Path(
                "./workflow_" + secrets.token_hex(3)
            ).absolute()  # should avoid side-effects. make dir later
        elif len(str(self.workflow_dir_path)) > 1:
            if not self.use_workflow_dir:
                self.__dict__["use_workflow_dir"] = True
            self.__dict__["workflow_dir_path"] = Path(self.workflow_dir_path).absolute()
        return self


class _EnsembleSpecs(BaseModel):
    """An all-encompassing model for a libEnsemble workflow."""

    H0: Optional[Any] = None  # np.ndarray - avoids sphinx issue
    """ A previous or preformatted libEnsemble History array to prepend. """

    libE_specs: LibeSpecs
    """ Specifications and options for libEnsemble. """

    sim_specs: SimSpecs
    """ Specifications for the simulation function. """

    gen_specs: Optional[GenSpecs] = None
    """ Specifications for the generator function. """

    exit_criteria: ExitCriteria
    """ Configurations for when to exit a workflow. """

    persis_info: Optional[dict] = None
    """ Per-worker information and structures to be passed between user function instances. """

    alloc_specs: Optional[AllocSpecs] = None
    """ Specifications for the allocation function. """

    nworkers: Optional[int] = None
    """ Number of worker processes to spawn (only in local/tcp modes). """
    # model_config = ConfigDict(arbitrary_types_allowed=True,
    # populate_by_name=True, extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def check_exit_criteria(self):
        return _check_exit_criteria(self)

    @model_validator(mode="after")
    def check_output_fields(self):
        return _check_output_fields(self)

    @model_validator(mode="after")
    def set_ensemble_nworkers(self):
        if self.libE_specs:
            self.nworkers = self.libE_specs.nworkers
        return self

    @model_validator(mode="after")
    def check_H0(self):
        if self.H0 is not None:
            return _check_H0(self)
        return self
