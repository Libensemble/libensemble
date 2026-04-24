import random
import warnings
from pathlib import Path

import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.utils.validators import (
    check_any_workers_and_disable_rm_if_tcp,
    check_exit_criteria,
    check_H0,
    check_input_dir_exists,
    check_inputs_exist,
    check_provided_ufuncs,
    check_set_gen_specs_from_variables,
    check_valid_comms_type,
    check_valid_in,
    check_valid_out,
    enable_save_H_when_every_K,
    set_calc_dirs_on_input_dir,
    set_default_comms,
    set_platform_specs_to_class,
    set_workflow_dir,
)

__all__ = ["SimSpecs", "GenSpecs", "AllocSpecs", "ExitCriteria", "LibeSpecs", "_EnsembleSpecs"]

# Deal with false warning https://github.com/pydantic/pydantic/issues/8677
if pydantic.__version__ == "2.6.0":
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings:")


def _get_dtype(field, name: str):
    """Get dtype from a VOCS field, handling discrete variables."""
    dtype = getattr(field, "dtype", None)
    # For discrete variables, infer dtype from values if not specified
    if dtype is None and hasattr(field, "values"):
        values = field.values
        if values:
            # Validate all values are the same type (required for NumPy array)
            value_types = {type(v) for v in values}
            if len(value_types) > 1:
                raise ValueError(
                    f"Discrete variable '{name}' has mixed types {value_types}. "
                    "All values must be the same type to be stored in NumPy array."
                )
            # Infer dtype from any value (all same type, scalar)
            # next(iter(values)) gets an element without creating a list
            sample_val = next(iter(values))
            if isinstance(sample_val, str):
                max_len = max(len(v) for v in values)
                dtype = f"U{max_len}"
            else:
                dtype = type(sample_val)
    return dtype


def _convert_dtype_to_output_tuple(name: str, dtype):
    """Convert dtype to proper output tuple format for NumPy dtype specification."""
    if dtype is None:
        dtype = float
    if isinstance(dtype, tuple):
        # Check if first element is a type (type, (shape,)) format
        if len(dtype) > 1 and (isinstance(dtype[0], type) or isinstance(dtype[0], str)):
            return (name, dtype[0], dtype[1])
        else:
            # Just shape (shape,) format, default to float
            return (name, float, dtype)
    else:
        return (name, dtype)


class SimSpecs(BaseModel):
    """
    Specifications for configuring a Simulation Function.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    sim_f: object = None
    """
    Python function matching the ``sim_f`` interface. Evaluates parameters
    produced by a generator function.
    """

    simulator: object | None = None
    """
    A callable (function) in gest-api format.
    When provided, ``sim_f`` defaults to the ``gest_api_sim`` wrapper.
    """

    inputs: list[str] | None = Field(default=[], alias="in")
    """
    list of **field names** out of the complete history to pass
    into the simulation function upon calling.
    """

    persis_in: list[str] | None = []
    """
    list of **field names** to send to a persistent simulation function
    throughout the run, following initialization.
    """

    # list of tuples for dtype construction
    outputs: list[tuple] = Field([], alias="out")
    """
    list of 2- or 3-tuples corresponding to NumPy dtypes.
    e.g. ``("dim", int, (3,))``, or ``("path", str)``.
    Typically used to initialize an output array within the simulation function:
    ``out = np.zeros(100, dtype=sim_specs["out"])``.
    Also necessary to construct libEnsemble's history array.
    """

    globus_compute_endpoint: str | None = ""
    """
    A Globus Compute (https://www.globus.org/compute) ID corresponding to an active endpoint on a remote system.
    libEnsemble's workers will submit simulator function instances to this endpoint instead of
    calling them locally.
    """

    threaded: bool | None = False
    """
    Instruct Worker process to launch user function to a thread.
    """

    user: dict | None = {}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters for customizing
    the simulator function.
    """

    vocs: object | None = None
    """
    A VOCS object. If provided and inputs/outputs are not explicitly set,
    they will be automatically derived from VOCS.
    """

    @field_validator("outputs")
    def check_valid_out(cls, v):
        return check_valid_out(cls, v)

    @field_validator("inputs", "persis_in")
    def check_valid_in(cls, v):
        return check_valid_in(cls, v)

    @model_validator(mode="after")
    def set_fields_from_vocs(self):
        """Set inputs and outputs from VOCS if vocs is provided and fields are not set."""
        # If simulator is provided but sim_f is not, default to gest_api_sim
        if self.simulator is not None and self.sim_f is None:
            from libensemble.sim_funcs.gest_api_wrapper import gest_api_sim

            self.sim_f = gest_api_sim

        if self.vocs is None:
            return self

        # Set inputs: variables + constants (what the sim receives)
        if not self.inputs:
            input_fields = []
            for attr in ["variables", "constants"]:
                if obj := getattr(self.vocs, attr, None):
                    input_fields.extend(list(obj.keys()))
            self.inputs = input_fields

        # Set outputs: objectives + observables + constraints (what the sim produces)
        if not self.outputs:
            out_fields = []
            for attr in ["objectives", "observables", "constraints"]:
                if obj := getattr(self.vocs, attr, None):
                    for name, field in obj.items():
                        dtype = getattr(field, "dtype", None)
                        out_fields.append(_convert_dtype_to_output_tuple(name, dtype))
            self.outputs = out_fields

        return self


class GenSpecs(BaseModel):
    """
    Specifications for configuring a Generator.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    generator: object | None = None
    """
    A pre-initialized generator object. Produces parameters for evaluation by a
    simulator function, and makes decisions based on simulator function output.

    These inherit from the `gest-api`
    (https://github.com/campa-consortium/gest-api) base class. Recommended over
    the classic ``gen_f`` interface.
    """

    gen_f: object | None = None
    """
    Python function matching the ``gen_f`` interface. Produces parameters for evaluation by a
    simulator function, and makes decisions based on simulator function output.
    """

    inputs: list[str] | None = Field(default=[], alias="in")
    """
    list of **field names** out of the complete history to pass
    into the generator function upon calling.
    """

    persis_in: list[str] | None = []
    """
    list of **field names** to send to a persistent generator function
    throughout the run, following initialization.
    """

    outputs: list[tuple] = Field([], alias="out")
    """
    list of 2- or 3-tuples corresponding to NumPy dtypes.
    e.g. ``("dim", int, (3,))``, or ``("path", str)``. Typically used to initialize an
    output array within the generator: ``out = np.zeros(100, dtype=gen_specs["out"])``.
    Also used to construct libEnsemble's history array.
    """

    initial_batch_size: int = 0
    """
    Initial sample size.
    For standardized generators, this is the number of initial points to request that the
    generator create. If zero, falls back to ``batch_size``.
    For persistent generators, this is the number of points evaluated before switching
    from batch return to asynchronous return (if ``async_return`` is True).

    Note: Certain generators included with libEnsemble decide batch sizes via
    ``gen_specs["user"]`` or other methods.
    """

    batch_size: int = 0
    """
    Number of points to generate in each batch. If zero, falls back to the number of
    completed evaluations most recently told to the generator.
    """

    initial_sample_method: str | None = None
    """
    Method for producing initial sample points before starting the generator.
    If None (default), the generator is responsible for producing its own initial
    sample via ``suggest()``. Set to ``"uniform"`` to have libEnsemble generate
    uniform random samples from VOCS bounds, evaluate them, and ingest the results
    into the generator before optimization begins. The number of sample points is
    determined by ``initial_batch_size``.
    """

    threaded: bool | None = False
    """
    Instruct Worker process to launch user function to a thread.
    """

    user: dict | None = {}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters for
    customizing the generator function
    """

    vocs: object | None = None
    """
    A VOCS object. If provided and persis_in/outputs are not explicitly set,
    they will be automatically derived from VOCS.
    """

    num_active_gens: int = 1
    """
    Maximum number of persistent generators to start.
    Only used if using the ``only_persistent_gens`` allocation function (the default).
    """

    async_return: bool = False
    """
    Return results to generator one-at-a-time as they come in (after sample). Default of False
    implies batch return.
    Only used if using the ``only_persistent_gens`` allocation function (the default).
    """

    active_recv_gen: bool = False
    """
    Initialize generator in active-receive mode. The generator can receive results
    even if it's not ready to produce new points.
    Only used if using the ``only_persistent_gens`` allocation function (the default).
    """

    batch_evaluate_same_priority: bool = False
    """
    Pass all points with the same priority value as a batch to a single simulator call.
    """

    alt_type: bool = False
    """
    Enable specialized allocator behavior for ``only_persistent_gens``.
    """

    @field_validator("outputs")
    def check_valid_out(cls, v):
        return check_valid_out(cls, v)

    @field_validator("inputs", "persis_in")
    def check_valid_in(cls, v):
        return check_valid_in(cls, v)

    @model_validator(mode="after")
    def set_fields_from_vocs(self):
        """Set persis_in and outputs from VOCS if vocs is provided and fields are not set."""
        if self.vocs is None:
            return self

        # Set persis_in: ALL VOCS fields (variables + constants + objectives + observables + constraints)
        if not self.persis_in:
            persis_in_fields = []
            for attr in ["variables", "constants", "objectives", "observables", "constraints"]:
                if obj := getattr(self.vocs, attr, None):
                    persis_in_fields.extend(list(obj.keys()))
            self.persis_in = persis_in_fields

        # Set inputs: same as persis_in for gest-api generators (needed for H0 ingestion)
        if not self.inputs and self.generator is not None:
            self.inputs = self.persis_in

        # Set outputs: variables + constants (what the generator produces)
        if not self.outputs:
            out_fields = []
            for attr in ["variables", "constants"]:
                if obj := getattr(self.vocs, attr, None):
                    for name, field in obj.items():
                        dtype = _get_dtype(field, name)
                        out_fields.append(_convert_dtype_to_output_tuple(name, dtype))
            self.outputs = out_fields

        # Add _id field if generator returns_id is True
        if self.generator is not None and getattr(self.generator, "returns_id", False):
            if self.outputs is None:
                self.outputs = []
            if "_id" not in [f[0] for f in self.outputs]:
                self.outputs.append(("_id", int))
            if self.persis_in is None:
                self.persis_in = []
            if "_id" not in self.persis_in:
                self.persis_in.append("_id")

        return self

    @model_validator(mode="after")
    def check_set_gen_specs_from_variables(self):
        return check_set_gen_specs_from_variables(self)


class AllocSpecs(BaseModel):
    """
    Specifications for configuring an Allocation Function.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    alloc_f: object = only_persistent_gens
    """
    Python function matching the ``alloc_f`` interface. Decides when simulator and generator functions
    should be called, and with what resources and parameters.

    .. note::
        For libEnsemble v2.0, the default allocation function is now ``only_persistent_gens``, instead
        of ``give_sim_work_first``.
    """

    user: dict | None = {}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters
    for customizing the allocation function.

    .. note::
        As of libEnsemble v2.0, options related to the default allocation function
        (e.g., ``async_return``, ``num_active_gens``) have been moved to
        :class:`GenSpecs<libensemble.specs.GenSpecs>`.
    """
    # end_alloc_tag


class ExitCriteria(BaseModel):
    """
    Specifications for configuring when libEnsemble should stop a given run.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    sim_max: int | None = None
    """Stop when this many new points have been evaluated by simulation functions."""

    gen_max: int | None = None
    """Stop when this many new points have been generated by generator functions."""

    wallclock_max: float | None = None
    """Stop when this many seconds has elapsed since the manager initialized."""

    stop_val: tuple[str, float] | None = None
    """Stop when ``H[str] < float`` for the given ``(str, float)`` pair."""


class LibeSpecs(BaseModel):
    """
    Specifications for configuring libEnsemble's runtime behavior.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    comms: str | None = "mpi"
    """
    Manager/Worker communications mode. ``'mpi'``, ``'local'``, ``'threads'``, or ``'tcp'``
    If ``nworkers`` is specified, then ``local`` comms will be used unless a parallel MPI
    environment is detected.
    """

    nworkers: int | None = 0
    """ Number of worker processes in ``"local"``, ``"threads"``, or ``"tcp"``."""

    gen_on_worker: bool = False
    """ Instructs libEnsemble to run generator functions on a worker rank.
    By default, the generator runs on the manager process as a thread (Worker 0).
    """

    mpi_comm: object | None = None
    """ libEnsemble MPI communicator. Default: ``MPI.COMM_WORLD``"""

    dry_run: bool | None = False
    """ Whether libEnsemble should immediately exit after validating all inputs. """

    abort_on_exception: bool | None = True
    """
    In MPI mode, whether to call ``MPI_ABORT`` on an exception.
    If ``False``, an exception will be raised by the manager.
    """

    save_every_k_sims: int | None = 0
    """ Save history array to file after every k evaluated points. """

    save_every_k_gens: int | None = 0
    """ Save history array to file after every k generated points. """

    save_H_and_persis_on_abort: bool | None = True
    """ Save states of ``H`` and ``persis_info`` to file on aborting after an exception."""

    save_H_on_completion: bool | None = False
    """
    Save state of ``H`` to file upon completing a workflow. Also enabled when either ``save_every_k_sims``
    or ``save_every_k_gens`` is set.
    """

    save_H_with_date: bool | None = False
    """ ``H`` filename contains date and timestamp."""

    H_file_prefix: str | None = "libE_history"
    """ Prefix for ``H`` filename."""

    worker_timeout: int | None = 1
    """ On libEnsemble shutdown, number of seconds after which workers considered timed out, then terminated. """

    kill_canceled_sims: bool | None = False
    """
    Try to kill sims with ``"cancel_requested"`` set ``True``.
    If ``False``, the manager avoids this moderate overhead.
    """

    use_workflow_dir: bool | None = False
    """
    Whether to place *all* log files, dumped arrays, and default output directories in a
    separate `workflow` directory. Each run will be suffixed with a hash.
    If copying back an ensemble directory from a scratch space, the copy is placed here.
    """

    reuse_output_dir: bool | None = False
    """
    Whether to allow overwrites and access to previous ensemble and workflow directories in subsequent runs.
    ``False`` by default to protect results.
    """

    workflow_dir_path: str | Path = "."
    """
    Optional path to the workflow directory.
    """

    ensemble_dir_path: str | Path | None = Path("ensemble")
    """
    Path to main ensemble directory. Can serve
    as a single working directory for workers, or contain calculation directories
    """

    ensemble_copy_back: bool | None = False
    """
    Whether to copy back contents of ``ensemble_dir_path`` to launch
    location. Useful if ``ensemble_dir_path`` is located on node-local storage.
    """

    use_worker_dirs: bool | None = False
    """ Whether to organize calculation directories under worker-specific directories. """

    sim_dirs_make: bool | None = False
    """
    Whether to make calculation directories for each simulation function call.
    """

    sim_dir_copy_files: list[str | Path] | None = []
    """
    Paths to copy into the working directory upon calling the simulation function.
    list of strings or ``pathlib.Path`` objects.
    """

    sim_dir_symlink_files: list[str | Path] | None = []
    """
    Paths to symlink into the working directory upon calling the simulation function.
    list of strings or ``pathlib.Path`` objects.
    """

    sim_input_dir: str | Path | None = None
    """
    Copy this directory's contents into the working directory upon calling the simulation function.
    Forms the base of a simulation directory.
    """

    gen_dirs_make: bool | None = False
    """
    Whether to make generator-specific calculation directories for each generator function call.
    """

    gen_dir_copy_files: list[str | Path] | None = []
    """
    Paths to copy into the working directory upon calling the generator function.
    list of strings or ``pathlib.Path`` objects
    """

    gen_dir_symlink_files: list[str | Path] | None = []
    """
    Paths to symlink into the working directory upon calling the generator function.
    list of strings or ``pathlib.Path`` objects.
    """

    gen_input_dir: str | Path | None = None
    """
    Copy this directory's contents into the working directory upon calling the generator function.
    Forms the base of a generator directory.
    """

    calc_dir_id_width: int | None = 4
    """
    The width of the numerical ID component of a calculation directory name. Leading
    zeros are padded to the sim/gen ID.
    """

    @field_validator("comms")
    def check_valid_comms_type(cls, value):
        return check_valid_comms_type(cls, value)

    @field_validator("platform_specs")
    def set_platform_specs_to_class(cls, value):
        return set_platform_specs_to_class(cls, value)

    @field_validator("sim_input_dir", "gen_input_dir")
    def check_input_dir_exists(cls, value):
        return check_input_dir_exists(cls, value)

    @field_validator("sim_dir_copy_files", "sim_dir_symlink_files", "gen_dir_copy_files", "gen_dir_symlink_files")
    def check_inputs_exist(cls, value):
        return check_inputs_exist(cls, value)

    @model_validator(mode="before")
    def set_default_comms(cls, values):
        return set_default_comms(cls, values)

    @model_validator(mode="after")
    def check_any_workers_and_disable_rm_if_tcp(self):
        return check_any_workers_and_disable_rm_if_tcp(self)

    @model_validator(mode="after")
    def enable_save_H_when_every_K(self):
        return enable_save_H_when_every_K(self)

    @model_validator(mode="after")
    def set_workflow_dir(self):
        return set_workflow_dir(self)

    @model_validator(mode="after")
    def set_calc_dirs_on_input_dir(self):
        return set_calc_dirs_on_input_dir(self)

    platform: str | None = ""
    """Name of a known platform defined in the platforms module.

    See :class:`Known Platforms list<libensemble.resources.platforms.Known_platforms>`.

    Example:

    .. code-block:: python

        libE_specs["platform"] = "perlmutter_g"

    Alternatively set the environment variable ``LIBE_PLATFORM``:

    .. code-block:: shell

        export LIBE_PLATFORM="perlmutter_g"

    See also option :attr:`platform_specs`.
    """

    platform_specs: object | dict | None = {}
    """A Platform object or dictionary specifying settings for a platform.

    To use existing platform:

    .. code-block:: python

        from libensemble.resources.platforms import PerlmutterGPU

        libE_specs["platform_specs"] = PerlmutterGPU()

    See :class:`Known Platforms list<libensemble.resources.platforms.Known_platforms>`.

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

    profile: bool | None = False
    """ Profile manager and worker logic using ``cProfile``. """

    disable_log_files: bool | None = False
    """ Disable ``ensemble.log`` and ``libE_stats.txt`` log files. """

    safe_mode: bool | None = False
    """ Prevents user functions from overwriting protected History fields, but requires moderate overhead. """

    stats_fmt: dict | None = {}
    """ Options for formatting ``'libE_stats.txt'``. See 'Formatting libE_stats.txt'. """

    live_data: object | None = None
    """ Add a live data capture object (e.g., for plotting). """

    workers: list[str] | None = []
    """ TCP Only: A list of worker hostnames. """

    ip: str | None = None
    """ TCP Only: IP address for Manager's system. """

    port: int | None = 0
    """ TCP Only: Port number for Manager's system. """

    authkey: str | None = f"libE_auth_{random.randrange(99999)}"
    """ TCP Only: Authkey for Manager's system."""

    workerID: int | None = None
    """ TCP Only: Worker ID number assigned to the new process. """

    worker_cmd: list[str] | None = []
    """
    TCP Only: Split string corresponding to worker/client Python process invocation. Contains
    a local Python path, user script, and manager/server format-fields for ``manager_ip``,
    ``manager_port``, ``authkey``, and ``workerID``. ``nworkers`` is specified normally.
    """

    use_persis_return_gen: bool | None = False
    """ Adds persistent generator output fields to the History array on return. """

    use_persis_return_sim: bool | None = False
    """ Adds persistent simulator output fields to the History array on return. """

    final_gen_send: bool | None = False
    """
    Send final simulation results to persistent generators before shutdown.
    The results will be sent along with the ``PERSIS_STOP`` tag.
    """

    disable_resource_manager: bool | None = False
    """
    Disable the built-in resource manager, including automatic resource detection
    and/or assignment of resources to workers. ``"resource_info"`` will be ignored.
    """

    num_resource_sets: int | None = 0
    """
    Total number of resource sets. Resources will be divided into this number.
    If not set, resources will be divided evenly by the number of workers.
    """

    gen_num_procs: int | None = 0
    """
    The default number of processors (MPI ranks) required by generators. Unless
    overridden by the equivalent `persis_info` settings, generators will be
    allocated this many processors for applications launched via the MPIExecutor.
    """

    gen_num_gpus: int | None = 0
    """
    The default number of GPUs required by generators. Unless overridden by
    the equivalent `persis_info` settings, generators will be allocated this
    many GPUs.
    """

    gpus_per_group: int | None = None
    """
    Number of GPUs for each group in the scheduler. This can be used to deal
    with scenarios where nodes have different numbers of GPUs. In effect a
    block of this many GPUs will be treated as a virtual node.
    By default the GPUs on a node are treated as a group.
    """

    use_tiles_as_gpus: bool | None = False
    """
    If ``True`` then treat a GPU tile as one GPU when GPU tiles is provided
    in platform specs or detected.
    """

    enforce_worker_core_bounds: bool | None = False
    """
    If ``False``, the Executor will permit the submission of tasks with a
    higher processor count than the CPUs available to the worker as
    detected by the resource manager. Larger node counts are not allowed.
    When ``"disable_resource_manager"`` is ``True``,
    this argument is ignored
    """

    dedicated_mode: bool | None = False
    """
    Instructs libEnsemble’s MPI executor not to run applications on nodes where
    libEnsemble processes (manager and workers) are running.
    """

    gen_workers: list[int] | None = []
    """
    list of workers that should only run generators. All other workers will only
    run simulator functions.
    """

    resource_info: dict | None = {}
    """
    Resource information to override automatically detected resources.
    Allowed fields are given below in 'Overriding Resource Auto-detection'.
    Note that if ``disable_resource_manager`` is set then this option is ignored.
    """

    scheduler_opts: dict | None = {}
    """ Options for the resource scheduler. See 'Scheduler Options' for more info """


class _EnsembleSpecs(BaseModel):
    """An all-encompassing model for a libEnsemble workflow."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    H0: object | None = None  # np.ndarray - avoids sphinx issue
    """ A previous or preformatted libEnsemble History array to prepend. """

    libE_specs: LibeSpecs
    """ Specifications and options for libEnsemble. """

    sim_specs: SimSpecs
    """ Specifications for the simulation function. """

    gen_specs: GenSpecs | None
    """ Specifications for the generator function. """

    exit_criteria: ExitCriteria
    """ Configurations for when to exit a workflow. """

    persis_info: dict | None = None
    """ Per-worker information and structures to be passed between user function instances. """

    alloc_specs: AllocSpecs | None = AllocSpecs()
    """ Specifications for the allocation function. """

    @model_validator(mode="after")
    def check_exit_criteria(self):
        return check_exit_criteria(self)

    @model_validator(mode="after")
    def check_H0(self):
        return check_H0(self)

    @model_validator(mode="after")
    def check_provided_ufuncs(self):
        return check_provided_ufuncs(self)
