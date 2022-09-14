import os
import random
import ipaddress
from typing import Dict, Callable, List, Any, Tuple, Union, Optional

import numpy as np
from pydantic import BaseModel, validator, root_validator, PyObject, Field, BaseConfig
from libensemble.tools.fields_keys import libE_fields

BaseConfig.arbitrary_types_allowed = True


class SimSpecs(BaseModel):
    sim_f: Callable
    inputs: List[str] = Field([], alias="in")
    persis_in: Optional[List[str]]
    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]] = []
    funcx_endpoint: Optional[str] = ""
    user: Optional[Dict]


class GenSpecs(BaseModel):
    gen_f: Callable
    inputs: Optional[List[str]] = Field([], alias="in")
    persis_in: Optional[List[str]]
    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]] = []
    funcx_endpoint: Optional[str] = ""
    user: Optional[Dict]


class AllocSpecs(BaseModel):
    alloc_f: Callable
    user: Optional[Dict]


class ExitCriteria(BaseModel):
    sim_max: Optional[int] = 100
    gen_max: Optional[int]
    wallclock_max: Optional[float]
    stop_val: Optional[Tuple[str, float]]

    @root_validator
    def check_any(cls, v):
        if not any(v.values()):
            raise ValueError("Must have some exit criterion")
        return v


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
    H_rows: Optional[np.ndarray]
    rset_team: Optional[List[int]] = None
    persistent: Optional[bool]
    gen_count: Optional[int]
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
    H_fields: List[str]
    persis_info: Dict
    tag: int
    libE_info: _LibEInfo


class _MPICommValidationModel:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    def validate(cls, comm):
        assert comm.Get_size() > 1, "Manager only - must be at least one worker (2 MPI tasks)"
        return comm


class LibeSpecs(BaseModel):
    abort_on_exception: Optional[bool] = True
    enforce_worker_core_bounds: Optional[bool] = False
    authkey: Optional[str] = f"libE_auth_{random.randrange(99999)}"
    disable_resource_manager: Optional[bool] = False
    dedicated_mode: Optional[bool] = False
    comms: str = "mpi"
    resource_info: Optional[ResourceInfo]
    disable_log_files: Optional[bool] = False
    final_fields: Optional[List[str]] = []
    ip: Optional[ipaddress.IPv4Address]
    kill_canceled_sims: Optional[bool] = True
    mpi_comm: Optional[_MPICommValidationModel] = None
    num_resource_sets: Optional[int]
    nworkers: Optional[int]
    port: Optional[int]
    profile: Optional[bool] = False
    safe_mode: Optional[bool] = True
    save_every_k_gens: Optional[int]
    save_every_k_sims: Optional[int]
    save_H_and_persis_on_abort: Optional[bool] = True
    scheduler_opts: Optional[SchedulerOpts] = {}
    stats_fmt: Optional[StatsFmt] = {}
    use_persis_return_gen: Optional[bool] = False
    use_persis_return_sim: Optional[bool] = False
    workerID: Optional[int]
    worker_timeout: Optional[int] = 1
    zero_resource_workers: Optional[List[int]]
    worker_cmd: Optional[List[str]]
    ensemble_copy_back: Optional[bool] = False
    ensemble_dir_path: Optional[str] = "./ensemble"
    use_worker_dirs: Optional[bool] = False
    sim_dirs_make: Optional[bool] = False
    sim_dir_copy_files: Optional[List[str]]
    sim_dir_symlink_files: Optional[List[str]]
    sim_input_dir: Optional[str]
    gen_dirs_make: Optional[bool] = False
    gen_dir_copy_files: Optional[List[str]]
    gen_dir_symlink_files: Optional[List[str]]
    gen_input_dir: Optional[str]

    class Config:
        arbitrary_types_allowed = True

    @validator("comms")
    def check_valid_comms_type(cls, value):
        assert value in ["mpi", "local", "tcp"], "Invalid comms type"
        return value

    @validator("sim_input_dir", "gen_input_dir")
    def check_input_dir_exists(cls, value):
        assert os.path.exists(value), "libE_specs['{}'] does not refer to an existing path.".format(value)
        return value

    @validator("sim_dir_copy_files", "sim_dir_symlink_files", "gen_dir_copy_files", "gen_dir_symlink_files")
    def check_inputs_exist(cls, value):
        for f in value:
            assert os.path.exists(f), "'{}' in libE_specs['{}'] does not refer to an existing path.".format(f, value)
        return value

    @root_validator
    def check_any_workers_and_disable_rm_if_tcp(cls, values):
        comms_type = values.get("comms")
        if comms_type in ["local", "tcp"]:
            assert values.get("nworkers") >= 1, "Must specify at least one worker"
        if comms_type == "tcp":
            values["disable_resource_manager"] = True  # Resource management not supported with TCP
        return values

    # @root_validator
    # def check_set_comm_world(cls, values):
    #     if not values.get("mpi_comm"):
    #         from mpi4py import MPI
    #         values["mpi_comm"] = MPI.COMM_WORLD
    #     return values


class Ensemble(BaseModel):
    H0: Optional[np.ndarray]
    libE_specs: LibeSpecs
    sim_specs: SimSpecs
    gen_specs: Optional[GenSpecs]
    exit_criteria: ExitCriteria
    persis_info: Optional[Dict]
    alloc_specs: Optional[AllocSpecs]
    nworkers: Optional[int]

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def check_exit_criteria(cls, values):
        if "stop_val" in values.get("exit_criteria"):
            stop_name = values.get("exit_criteria").stop_val[0]
            sim_out_names = [e[0] for e in values.get("sim_specs").out]
            gen_out_names = [e[0] for e in values.get("gen_specs").out]
            assert (
                stop_name in sim_out_names + gen_out_names
            ), "Can't stop on {} if it's not in a sim/gen output".format(stop_name)
        return values

    @root_validator
    def check_output_fields(cls, values):
        out_names = [e[0] for e in libE_fields]
        if values.get("H0") and values.get("H0").dtype.names is not None:
            out_names += list(values.get("H0").dtype.names)
        out_names += [e[0] for e in values.get("sim_specs").out]
        if values.get("gen_specs"):
            out_names += [e[0] for e in values.get("gen_specs").out]

        for name in values.get("libE_specs").final_fields:
            assert name in out_names, (
                name + " in libE_specs['fields_keys'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )

        for name in values.get("sim_specs").inputs:
            assert name in out_names, (
                name + " in sim_specs['in'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )

        if values.get("gen_specs"):
            for name in values.get("gen_specs").inputs:
                assert name in out_names, (
                    name + " in gen_specs['in'] is not in sim_specs['out'], "
                    "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
                )
        return values

    @root_validator
    def set_ensemble_nworkers(cls, values):
        if values.get("libE_specs"):
            values["nworkers"] = values["libE_specs"].nworkers
            return values

    @root_validator
    def check_H0(cls, values):
        if len(values.get("H0")) > 0:
            H0 = values.get("H0")
            specs = [values.get("sim_specs"), values.get("alloc_specs"), values.get("gen_specs")]
            dtype_list = list(set(libE_fields + sum([k.out or [] for k in specs if k], [])))
            Dummy_H = np.zeros(1 + len(H0), dtype=dtype_list)

            fields = H0.dtype.names

            assert set(fields).issubset(set(Dummy_H.dtype.names)), "H0 contains fields {} not in the History.".format(
                set(fields).difference(set(Dummy_H.dtype.names))
            )

            assert "sim_ended" not in fields or np.all(
                H0["sim_started"] == H0["sim_ended"]
            ), "H0 contains unreturned or invalid points"

            def _check_consistent_field(name, field0, field1):
                """Checks that new field (field1) is compatible with an old field (field0)."""
                assert field0.ndim == field1.ndim, "H0 and H have different ndim for field {}".format(name)
                assert np.all(
                    np.array(field1.shape) >= np.array(field0.shape)
                ), "H too small to receive all components of H0 in field {}".format(name)

            for field in fields:
                 _check_consistent_field(field, H0[field], Dummy_H[field])
        return values