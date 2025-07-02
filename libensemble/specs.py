import random
import warnings
from pathlib import Path

import pydantic
from pydantic import BaseModel, Field

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first

__all__ = ["SimSpecs", "GenSpecs", "AllocSpecs", "ExitCriteria", "LibeSpecs", "_EnsembleSpecs"]

# Deal with false warning https://github.com/pydantic/pydantic/issues/8677
if pydantic.__version__ == "2.6.0":
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings:")


"""
Pydantic-version agnostic
"""


class SimSpecs(BaseModel):
    """
    Specifications for configuring a Simulation Function.
    """

    sim_f: object = None
    """
    Python function matching the ``sim_f`` interface. Evaluates parameters
    produced by a generator function.
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


class GenSpecs(BaseModel):
    """
    Specifications for configuring a Generator Function.
    """

    gen_f: object | None = None
    """
    Python function matching the ``gen_f`` interface. Produces parameters for evaluation by a
    simulator function, and makes decisions based on simulator function output.
    """

    generator: object | None = None
    """
    A pre-initialized generator object.
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

    globus_compute_endpoint: str | None = ""
    """
    A Globus Compute (https://www.globus.org/compute) ID corresponding to an active endpoint on a remote system.
    libEnsemble's workers will submit generator function instances to this endpoint instead of
    calling them locally.
    """

    initial_batch_size: int = 0
    """
    Number of initial points to request that the generator create. If zero, falls back to ``batch_size``.
    If both options are zero, defaults to the number of workers.

    Note: Certain generators included with libEnsemble decide
    batch sizes via ``gen_specs["user"]`` or other methods.
    """

    batch_size: int = 0
    """
    Number of points to generate in each batch. If zero, falls back to the number of
    completed evaluations most recently told to the generator.

    Note: Certain generators included with libEnsemble decide
    batch sizes via ``gen_specs["user"]`` or other methods.
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


class AllocSpecs(BaseModel):
    """
    Specifications for configuring an Allocation Function.
    """

    alloc_f: object = give_sim_work_first
    """
    Python function matching the ``alloc_f`` interface. Decides when simulator and generator functions
    should be called, and with what resources and parameters.
    """

    user: dict | None = {"num_active_gens": 1}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters
    for customizing the allocation function.
    """

    outputs: list[tuple] = Field([], alias="out")
    """
    list of 2- or 3-tuples corresponding to NumPy dtypes. e.g. ``("dim", int, (3,))``, or ``("path", str)``.
    Allocation functions that modify libEnsemble's History array with additional fields should list those
    fields here. Also used to construct libEnsemble's history array.
    """
    # end_alloc_tag


class ExitCriteria(BaseModel):
    """
    Specifications for configuring when libEnsemble should stop a given run.
    """

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

    comms: str | None = "mpi"
    """
    Manager/Worker communications mode. ``'mpi'``, ``'local'``, ``'threads'``, or ``'tcp'``
    If ``nworkers`` is specified, then ``local`` comms will be used unless a parallel MPI
    environment is detected.
    """

    nworkers: int | None = 0
    """ Number of worker processes in ``"local"``, ``"threads"``, or ``"tcp"``."""

    gen_on_manager: bool | None = False
    """ Instructs Manager process to run generator functions.
    This generator function can access/modify user objects by reference.
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

    workflow_dir_path: str | Path | None = "."
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
    a local Python path, calling script, and manager/server format-fields for ``manager_ip``,
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
    If not set, resources will be divided evenly (excluding zero_resource_workers).
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

    zero_resource_workers: list[int] | None = []
    """
    list of workers that require no resources. For when a fixed mapping of workers
    to resources is required. Otherwise, use ``num_resource_sets``.
    For use with supported allocation functions.
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


def input_fields(fields: list[str]):
    """Decorates a user-function with a list of field names to pass in on initialization.

    Decorated functions don't need those fields specified in ``SimSpecs.inputs`` or ``GenSpecs.inputs``.

    .. code-block:: python

        from libensemble.specs import input_fields, output_data


        @input_fields(["x"])
        @output_data([("f", float)])
        def norm_eval(x, persis_info, sim_specs):
            H_o = np.zeros(1, dtype=sim_specs["out"])
            H_o["f"] = np.linalg.norm(x)
            return H_o, persis_info
    """

    def decorator(func):
        setattr(func, "inputs", fields)
        if not func.__doc__:
            func.__doc__ = ""
        func.__doc__ = f"\n    **Input Fields:** ``{func.inputs}``\n" + func.__doc__
        return func

    return decorator


def persistent_input_fields(fields: list[str]):
    """Decorates a *persistent* user-function with a list of field names to send in throughout runtime.

    Decorated functions don't need those fields specified in ``SimSpecs.persis_in`` or ``GenSpecs.persis_in``.

    .. code-block:: python

        from libensemble.specs import persistent_input_fields, output_data


        @persistent_input_fields(["f"])
        @output_data(["x", float])
        def persistent_uniform(_, persis_info, gen_specs, libE_info):

            b, n, lb, ub = _get_user_params(gen_specs["user"])
            ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

            tag = None
            while tag not in [STOP_TAG, PERSIS_STOP]:
                H_o = np.zeros(b, dtype=gen_specs["out"])
                H_o["x"] = persis_info["rand_stream"].uniform(lb, ub, (b, n))
                tag, Work, calc_in = ps.send_recv(H_o)
                if hasattr(calc_in, "__len__"):
                    b = len(calc_in)

            return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
    """

    def decorator(func):
        setattr(func, "persis_in", fields)
        if not func.__doc__:
            func.__doc__ = ""
        func.__doc__ = f"\n    **Persistent Input Fields:** ``{func.persis_in}``\n" + func.__doc__
        return func

    return decorator


def output_data(fields: list[tuple]):
    """Decorates a user-function with a list of tuples corresponding to NumPy dtypes for the function's output data.

    Decorated functions don't need those fields specified in ``SimSpecs.outputs`` or ``GenSpecs.outputs``.

    .. code-block:: python

        from libensemble.specs import input_fields, output_data


        @input_fields(["x"])
        @output_data([("f", float)])
        def norm_eval(x, persis_info, sim_specs):
            H_o = np.zeros(1, dtype=sim_specs["out"])
            H_o["f"] = np.linalg.norm(x)
            return H_o, persis_info
    """

    def decorator(func):
        setattr(func, "outputs", fields)
        if not func.__doc__:
            func.__doc__ = ""
        func.__doc__ = f"\n    **Output Datatypes:** ``{func.outputs}``\n" + func.__doc__
        return func

    return decorator
