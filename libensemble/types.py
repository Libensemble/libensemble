import os
import random
import ipaddress
from typing import Dict, Callable, List, Any, Tuple, Union, Optional

import numpy as np
from pydantic import BaseModel, validator, root_validator, PyObject, Field
from libensemble.tools.fields_keys import libE_fields


class SimSpecs(BaseModel):
    sim_f: Callable
    inputs: List[str] = Field(alias="in")
    persis_in: Optional[List[str]]
    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]]
    funcx_endpoint: Optional[str]
    user: Optional[Dict]


class GenSpecs(BaseModel):
    gen_f: Callable
    inputs: Optional[List[str]] = Field(alias="in")
    persis_in: Optional[List[str]]
    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]]
    funcx_endpoint: Optional[str]
    user: Optional[Dict]


class AllocSpecs(BaseModel):
    alloc_f: Callable
    user: Optional[Dict]


class ExitCriteria(BaseModel):
    sim_max: Optional[int]
    gen_max: Optional[int]
    wallclock_max: Optional[float]
    stop_val: Optional[Tuple[str, float]]

    @root_validator
    def check_any(cls, v):
        if not any(v.values()):
            raise ValueError("Must have some exit criterion")


class ResourceInfo(BaseModel):
    cores_on_node: Optional[Tuple[int, int]]
    node_file: Optional[str]
    nodelist_env_slurm: Optional[str]
    nodelist_env_cobalt: Optional[str]
    nodelist_env_lsf: Optional[str]
    nodelist_env_lsf_shortform: Optional[str]


class SchedulerOpts(BaseModel):
    split2fit: Optional[bool] = True
    match_slots: Optional[bool] = True


class StatsFmt(BaseModel):
    task_timing: Optional[bool] = False
    task_datetime: Optional[bool] = False
    show_resource_sets: Optional[bool] = False


class _LibEInfo(BaseModel):
    rset_team: Optional[List[int]] = None
    gen_count: Optional[int]
    H_rows: np.ndarray
    persistent: Optional[bool]
    active_recv: Optional[bool]
    workerID: Optional[int]
    comm: Optional[PyObject]

    class Config:
        arbitrary_types_allowed = True


class _AllocInfo(BaseModel):
    exit_criteria: ExitCriteria
    elapsed_time: float
    manager_kill_canceled_sims: bool
    sim_started_count: int
    sim_ended_count: int
    gen_informed_count: int
    sim_max_given: bool
    use_resource_sets: bool


class _Work(BaseModel):
    persis_info: Dict
    H_fields: List[str]
    tag: int
    libE_info: Dict
    H_rows: Optional[List[int]]
    blocking: Optional[List[int]]
    persistent: Optional[bool]


class LibeSpecs(BaseModel):
    abort_on_exception: Optional[bool] = True
    enforce_worker_core_bounds: Optional[bool] = False
    authkey: Optional[str] = f"libE_auth_{random.randrange(99999)}"
    disable_resource_manager: Optional[bool] = False
    dedicated_mode: Optional[bool] = False
    comms: str = "mpi"
    resource_info: Optional[ResourceInfo]
    disable_log_files: Optional[bool] = False
    final_fields: Optional[List[str]]
    ip: Optional[ipaddress.IPv4Address]
    kill_canceled_sims: Optional[bool] = True
    mpi_comm: Optional[PyObject] = None
    num_resource_sets: Optional[int]
    nworkers: int
    port: Optional[int]
    profile: Optional[bool] = False
    safe_mode: Optional[bool] = True
    save_every_k_gens: Optional[int]
    save_every_k_sims: Optional[int]
    save_H_and_persis_on_abort: Optional[bool] = True
    scheduler_opts: Optional[SchedulerOpts]
    stats_fmt: Optional[StatsFmt]
    use_persis_return_gen: Optional[bool] = False
    use_persis_return_sim: Optional[bool] = False
    workerID: Optional[int]
    worker_timeout: Optional[int] = 1
    zero_resource_workers: Optional[List[int]]
    worker_cmd: Optional[List[str]]
    ensemble_copy_back: Optional[bool] = False
    ensemble_dir_path: Optional[str]
    use_worker_dirs: Optional[bool] = False
    sim_dirs_make: Optional[bool] = False
    sim_dir_copy_files: Optional[List[str]]
    sim_dir_symlink_files: Optional[List[str]]
    sim_input_dir: Optional[str]
    gen_dirs_make: Optional[bool] = False
    gen_dir_copy_files: Optional[List[str]]
    gen_dir_symlink_files: Optional[List[str]]
    gen_input_dir: Optional[str]

    @root_validator
    def check_not_manager_only(cls, values):
        if values.get("comms") == "mpi":
            assert values.get("mpi_comm").Get_size() > 1, "Manager only - must be at least one worker (2 MPI tasks)"

    @root_validator
    def check_any_workers(cls, values):
        if values.get("comms") in ["local", "tcp"]:
            assert values.get("nworkers") >= 1, "Must specify at least one worker"

    @validator("comms")
    def check_valid_comms_type(cls, value):
        assert value in ["mpi", "local", "tcp"], "Invalid comms type"

    @validator("sim_input_dir", "gen_input_dir")
    def check_input_dir_exists(cls, value):
        assert os.path.exists(value), "libE_specs['{}'] does not refer to an existing path.".format(value)

    @validator("sim_dir_copy_files", "sim_dir_symlink_files", "gen_dir_copy_files", "gen_dir_symlink_files")
    def check_inputs_exist(cls, value):
        for f in value:
            assert os.path.exists(f), "'{}' in libE_specs['{}'] does not refer to an existing path.".format(f, value)


class Ensemble(BaseModel):
    H0: Optional[np.ndarray] = None
    libE_specs: LibeSpecs
    persis_info: Optional[Dict]
    sim_specs: SimSpecs
    gen_specs: Optional[GenSpecs]
    alloc_specs: Optional[AllocSpecs]
    exit_criteria: ExitCriteria

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def check_exit_criteria(cls, values):
        if "stop_val" in values.get("exit_criteria"):
            stop_name = values.get("exit_criteria")["stop_val"][0]
            sim_out_names = [e[0] for e in values.get("sim_specs")["out"]]
            gen_out_names = [e[0] for e in values.get("gen_specs")["out"]]
            assert (
                stop_name in sim_out_names + gen_out_names
            ), "Can't stop on {} if it's not in a sim/gen output".format(stop_name)

    @root_validator
    def check_output_fields(cls, values):
        out_names = [e[0] for e in libE_fields]
        if values.get("H0") and values.get("H0").dtype.names is not None:
            out_names += list(values.get("H0").dtype.names)
        out_names += [e[0] for e in values.get("sim_specs").get("out", [])]
        if values.get("gen_specs"):
            out_names += [e[0] for e in values.get("gen_specs").get("out", [])]

        for name in values.get("libE_specs").get("final_fields", []):
            assert name in out_names, (
                name + " in libE_specs['fields_keys'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )

        for name in values.get("sim_specs").get("in", []):
            assert name in out_names, (
                name + " in sim_specs['in'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )

        if values.get("gen_specs"):
            for name in values.get("gen_specs").get("in", []):
                assert name in out_names, (
                    name + " in gen_specs['in'] is not in sim_specs['out'], "
                    "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
                )

    @root_validator
    def check_H0(cls, values):
        if values.get("H0") and len(values.get("H0")):
            pass  # TODO: finish
