import os
import random
import secrets
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseConfig, BaseModel, Field, root_validator, validator

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.gen_funcs.sampling import latin_hypercube_sample
from libensemble.resources.platforms import Platform
from libensemble.sim_funcs.one_d_func import one_d_example
from libensemble.utils.specs_checkers import (
    MPI_Communicator,
    _check_any_workers_and_disable_rm_if_tcp,
    _check_exit_criteria,
    _check_H0,
    _check_output_fields,
)

_UNRECOGNIZED_ERR = "Unrecognized field. Check closely for typos, or libEnsemble's docs"
_OUT_DTYPE_ERR = "Unable to coerce '{}' into a NumPy dtype. It should be a list of 2-tuples or 3-tuples"
_IN_INVALID_ERR = "Value should be a list of field names (a list of strings)"
_UFUNC_INVALID_ERR = "Specified sim_f or gen_f is not callable. It should be a user function"

BaseConfig.arbitrary_types_allowed = True
BaseConfig.allow_population_by_field_name = True
BaseConfig.extra = "forbid"
BaseConfig.error_msg_templates = {
    "value_error.extra": _UNRECOGNIZED_ERR,
    "type_error.callable": _UFUNC_INVALID_ERR,
}
BaseConfig.validate_assignment = True

__all__ = ["SimSpecs", "GenSpecs", "AllocSpecs", "ExitCriteria", "LibeSpecs", "EnsembleSpecs"]


class SimSpecs(BaseModel):
    """
    Specifications for configuring a Simulation Function. Equivalent to
    a ``sim_specs`` dictionary.
    """

    sim_f: Callable = one_d_example
    """
    Python function that matches the ``sim_f`` api. e.g. ``libensemble.sim_funcs.borehole``. Evaluates parameters
    produced by a generator function
    """

    inputs: List[str] = Field([], alias="in")
    """
    List of field names out of the complete history to pass
    into the simulation function on initialization. Can use ``in`` or ``inputs`` as keyword.
    """

    persis_in: Optional[List[str]] = []
    """
    List of field names that will be passed to a persistent simulation function
    throughout runtime, following initialization
    """

    # list of tuples for dtype construction
    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]] = []
    """
    List of tuples corresponding to NumPy dtypes. e.g. ``("dim", int, (3,))``, or ``("path", str)``.
    Typically used to initialize an output array within the simulation function:
    ``out = np.zeros(100, dtype=sim_specs["out"])``.
    Also used to construct the complete dtype for libEnsemble's history array
    """

    funcx_endpoint: Optional[str] = ""
    """
    A funcX (https://funcx.org/) ID corresponding to an active endpoint on a remote system. libEnsemble's workers
    will submit simulator function instances to this endpoint to be executed, instead of calling them locally
    """

    user: Optional[dict] = {}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters for customizing
    the simulator function
    """

    @validator("out", pre=True)
    def check_valid_out(cls, v):
        try:
            _ = np.dtype(v)
        except TypeError:
            raise ValueError(_OUT_DTYPE_ERR.format(v))
        else:
            return v

    @validator("inputs", "persis_in", pre=True)
    def check_valid_in(cls, v):
        if not all(isinstance(s, str) for s in v):
            raise ValueError(_IN_INVALID_ERR)
        return v


class GenSpecs(BaseModel):
    """
    Specifications for configuring a Generator Function. Equivalent to
    a ``gen_specs`` dictionary.
    """

    gen_f: Optional[Callable] = latin_hypercube_sample
    """
    Python function that matches the gen_f api. e.g. `libensemble.gen_funcs.sampling`. Produces parameters for
    evaluation by a simulator function, and makes decisions based on simulator function output
    """

    inputs: Optional[List[str]] = Field([], alias="in")
    """
    List of field names out of the complete history to pass
    into the simulation function on initialization. Can use ``in`` or ``inputs`` as keyword
    """

    persis_in: Optional[List[str]] = []
    """
    List of field names that will be passed to a persistent generator function
    throughout runtime, following initialization
    """

    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]] = []
    """
    List of tuples corresponding to NumPy dtypes. e.g. ``("dim", int, (3,))``, or ``("path", str)``.
    Typically used to initialize an output array within the generator function:
    ``out = np.zeros(100, dtype=gen_specs["out"])``. Also used to construct the complete dtype for libEnsemble's
    history array
    """

    funcx_endpoint: Optional[str] = ""
    """
    A funcX (https://funcx.org/) ID corresponding to an active endpoint on a remote system. libEnsemble's workers
    will submit generator function instances to this endpoint to be executed, instead of being called in-place
    """

    user: Optional[dict] = {}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters for customizing the generator
    function
    """

    @validator("out", pre=True)
    def check_valid_out(cls, v):
        try:
            _ = np.dtype(v)
        except TypeError:
            raise ValueError(_OUT_DTYPE_ERR.format(v))
        else:
            return v

    @validator("inputs", "persis_in", pre=True)
    def check_valid_in(cls, v):
        if not all(isinstance(s, str) for s in v):
            raise ValueError(_IN_INVALID_ERR)
        return v


class AllocSpecs(BaseModel):
    """
    Specifications for configuring an Allocation Function. Equivalent to
    an ``alloc_specs`` dictionary.
    """

    alloc_f: Callable = give_sim_work_first
    """
    Python function that matches the alloc_f api. e.g. `libensemble.alloc_funcs.give_sim_work_first`. Decides if and
    when simulator and generator functions should be called, and with what resources and parameters
    """

    user: Optional[dict] = {"num_active_gens": 1}
    """
    A user-data dictionary to place bounds, constants, settings, or other parameters for customizing the allocation
    function
    """

    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]] = []
    """
    List of tuples corresponding to NumPy dtypes. e.g. ``("dim", int, (3,))``, or ``("path", str)``.
    Allocation functions that modify libEnsemble's History array with additional fields (e.g. to mark
    timing information, or determine if parameters should be distributed again, etc.) should list those
    fields here. Also used to construct the complete dtype for libEnsemble's history array
    """
    # end_alloc_tag


class ExitCriteria(BaseModel):
    """
    Specifications for configuring when libEnsemble should stop a given run. Equivalent to an
    ``exit_criteria`` dictionary.
    """

    sim_max: Optional[int]
    """ Stop when this many new points have been evaluated by simulation functions"""

    gen_max: Optional[int]
    """Stop when this many new points have been generated by generator functions"""

    wallclock_max: Optional[float]
    """Stop when this much time (in seconds) has elapsed since the manager initialized"""

    stop_val: Optional[Tuple[str, float]]
    """Stop when ``H[str] < float`` for the given ``(str, float)`` pair"""


class LibeSpecs(BaseModel):
    """
    Specifications for configuring libEnsemble's runtime behavior. Equivalent to a ``libE_specs`` dictionary.
    """

    comms: Optional[str] = "mpi"
    """ Manager/Worker communications mode. ``'mpi'``, ``'local'``, or ``'tcp'`` """

    nworkers: Optional[int]
    """ Number of worker processes to spawn (only in local/tcp modes) """

    mpi_comm: Optional[MPI_Communicator] = None  # see utils/specs_checkers.py
    """ libEnsemble communicator. Default: ``MPI.COMM_WORLD`` """

    dry_run: Optional[bool] = False
    """ Whether libEnsemble should immediately exit after validating all inputs """

    abort_on_exception: Optional[bool] = True
    """
    In MPI mode, whether to call ``MPI_ABORT`` on an exception.
    If False, an exception will be raised by the manager
    """

    save_every_k_sims: Optional[int] = 0
    """ Save history array to file after every k evaluated points """

    save_every_k_gens: Optional[int] = 0
    """  Save history array to file after every k generated points """

    save_H_and_persis_on_abort: Optional[bool] = True
    """ Save states of ``H`` and ``persis_info`` on aborting after an exception"""

    worker_timeout: Optional[int] = 1
    """ On libEnsemble shutdown, number of seconds after which workers considered timed out, then terminated """

    kill_canceled_sims: Optional[bool] = True
    """
    Instructs libEnsemble to send kill signals to sims with their ``cancel_requested`` field set.
    If ``False``, the manager avoids this moderate overhead
    """

    use_workflow_dir: Optional[bool] = False
    """
    Whether to place *all* log files, dumped arrays, and default ensemble-directories in a
    separate `workflow` directory. New runs and their workflow directories will be automatically
    differentiated. If copying back an ensemble directory from a scratch space, the copy is placed
    in the workflow directory.
    """

    workflow_dir_path: Optional[Union[str, Path]] = "."
    """
    Optional path to the workflow directory. Autogenerated in the current directory if `use_workflow_dir`
    is specified.
    """

    ensemble_dir_path: Optional[str] = "ensemble"
    """
    Path to main ensemble directory containing calculation directories. Can serve
    as single working directory for workers, or contain calculation directories
    """

    ensemble_copy_back: Optional[bool] = False
    """
    Whether to copy back directories within ``ensemble_dir_path`` back to launch
    location. Useful if ensemble directory placed on node-local storage
    """

    use_worker_dirs: Optional[bool] = False
    """ Whether to organize calculation directories under worker-specific directories """

    sim_dirs_make: Optional[bool] = False
    """
    Whether to make simulation-specific calculation directories for each simulation function call.
    By default all workers operate within the top-level ensemble directory
    """

    sim_dir_copy_files: Optional[List[str]] = []
    """ Paths to files or directories to copy into each simulation or ensemble directory """

    sim_dir_symlink_files: Optional[List[str]] = []
    """ Paths to files or directories to symlink into each simulation directory """

    sim_input_dir: Optional[str] = ""
    """
    Copy this directory and its contents for each simulation-specific directory.
    If not using calculation directories, contents are copied to the ensemble directory
    """

    gen_dirs_make: Optional[bool] = False
    """
    Whether to make generator-specific calculation directories for each generator function call.
    By default all workers operate within the top-level ensemble directory
    """

    gen_dir_copy_files: Optional[List[str]] = []
    """ Paths to files or directories to copy into each generator or ensemble directory """

    gen_dir_symlink_files: Optional[List[str]] = []
    """ Paths to files or directories to symlink into each generator directory """

    gen_input_dir: Optional[str] = ""
    """
    Copy this directory and its contents for each generator-instance-specific directory.
    If not using calculation directories, contents are copied to the ensemble directory
    """

    platform: Optional[str] = ""
    """Name of a known platform defined in the platforms module."""

    platform_spec : Optional[Union[Platform, dict]] = {}
    """A Platform obj or dictionary specifying settings for a platform."""

    profile: Optional[bool] = False
    """ Profile manager and worker logic using cProfile """

    disable_log_files: Optional[bool] = False
    """ Disable the creation of ``ensemble.log`` and ``libE_stats.txt`` log files """

    safe_mode: Optional[bool] = True
    """ Prevents user functions from overwriting protected History fields, but requires moderate overhead """

    stats_fmt: Optional[dict] = {}
    """ Options for formatting 'libE_stats.txt'. See 'Formatting Options for libE_stats File' for more info """

    workers: Optional[List[str]]
    """ TCP Only: A list of worker hostnames """

    ip: Optional[str] = None
    """ TCP Only: IP address for Manager's system """

    port: Optional[int] = 0
    """ TCP Only: Port number for Manager's system """

    authkey: Optional[str] = f"libE_auth_{random.randrange(99999)}"
    """ TCP Only: Authkey for Manager's system"""

    workerID: Optional[int]
    """ TCP Only: Worker ID number assigned to the new process """

    worker_cmd: Optional[List[str]]
    """
    TCP Only: Split string corresponding to worker/client Python process invocation. Contains
    a local Python path, calling script, and manager/server format-fields for manager_ip,
    manager_port, authkey, and workerID. nworkers is specified normally
    """

    use_persis_return_gen: Optional[bool] = False
    """ Adds persistent generator output fields to the History array on return """

    use_persis_return_sim: Optional[bool] = False
    """ Adds persistent simulator output fields to the History array on return """

    final_fields: Optional[List[str]] = []
    """
    List of fields in ``H`` that the manager will return to persistent
    workers along with the ``PERSIS_STOP`` tag at the end of a run
    """

    disable_resource_manager: Optional[bool] = False
    """
    Disable the built-in resource manager. If ``True``, automatic resource detection
    and/or assignment of resources to workers is disabled. ``resource_info`` will
    also be ignored
    """

    num_resource_sets: Optional[int]
    """
    Total number of resource sets. Resources will be divided into this number.
    If not set, resources will be divided evenly (excluding zero_resource_workers).
    """

    enforce_worker_core_bounds: Optional[bool] = False
    """
    If ``False``, the Executor will permit submission of tasks with a
    higher processor count than the CPUs available to the worker as
    detected by the resource manager. Larger node counts are not allowed.
    When ``"disable_resource_manager"`` is ``True``,
    this argument is ignored
    """

    dedicated_mode: Optional[bool] = False
    """
    Instructs libEnsemble to not run applications on resources where libEnsemble
    processes (manager and workers) are running
    """

    zero_resource_workers: Optional[List[int]] = []
    """
    List of workers that require no resources. For when a fixed mapping of workers
    to resources is required. Otherwise, use ``num_resource_sets``
    For use with supported allocation functions
    """

    resource_info: Optional[dict] = {}
    """
    Resource information to override automatically detected resources.
    Allowed fields are given below in 'Overriding Auto-detection'
    Note that if ``disable_resource_manager`` is set then this option is ignored
    """

    scheduler_opts: Optional[dict] = {}
    """ Options for the resource scheduler. See 'Scheduler Options' for more info """


    class Config:
        arbitrary_types_allowed = True

    @validator("comms")
    def check_valid_comms_type(cls, value: str) -> str:
        assert value in ["mpi", "local", "tcp"], "Invalid comms type"
        return value

    @validator("sim_input_dir", "gen_input_dir")
    def check_input_dir_exists(cls, value: str) -> str:
        if len(value):
            assert os.path.exists(value), "libE_specs['{}'] does not refer to an existing path.".format(value)
        return value

    @validator("sim_dir_copy_files", "sim_dir_symlink_files", "gen_dir_copy_files", "gen_dir_symlink_files")
    def check_inputs_exist(cls, value: List[str]) -> List[str]:
        for f in value:
            assert os.path.exists(f), "'{}' in libE_specs['{}'] does not refer to an existing path.".format(f, value)
        return value

    @root_validator
    def check_any_workers_and_disable_rm_if_tcp(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return _check_any_workers_and_disable_rm_if_tcp(values)

    @root_validator
    def set_defaults_on_mpi(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("comms") == "mpi":
            from mpi4py import MPI

            if values.get("mpi_comm") is None:  # not values.get("mpi_comm") is True ???
                values["mpi_comm"] = MPI.COMM_WORLD
        return values

    @root_validator
    def set_workflow_dir(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("use_workflow_dir") and len(str(values.get("workflow_dir_path"))) <= 1:
            values["workflow_dir_path"] = Path(
                "./workflow_" + secrets.token_hex(3)
            ).absolute()  # should avoid side-effects. make dir later
        elif len(str(values.get("workflow_dir_path"))) > 1 and not values.get("use_workflow_dir"):
            values["use_workflow_dir"] = True
        return values


class EnsembleSpecs(BaseModel):
    """An all-encompasing model for a libEnsemble workflow."""

    H0: Optional[Any] = None  # np.ndarray - avoids sphinx issue
    """ A previous or preformatted libEnsemble History array to prepend """

    libE_specs: LibeSpecs
    """ Specifications and options for libEnsemble """

    sim_specs: SimSpecs
    """ Specifications for the simulation function """

    gen_specs: Optional[GenSpecs]
    """ Specifications for the generator function """

    exit_criteria: ExitCriteria
    """ Configurations for when to exit a workflow """

    persis_info: Optional[dict]
    """ Per-worker information and structures to be passed between user function instances """

    alloc_specs: Optional[AllocSpecs]
    """ Specifications for the allocation function """

    nworkers: Optional[int]
    """ Number of worker processes to spawn (only in local/tcp modes) """

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def check_exit_criteria(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return _check_exit_criteria(values)

    @root_validator
    def check_output_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return _check_output_fields(values)

    @root_validator
    def set_ensemble_nworkers(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("libE_specs"):
            values["nworkers"] = values["libE_specs"].nworkers
        return values

    @root_validator
    def check_H0(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("H0") is not None:
            return _check_H0(values)
        return values
