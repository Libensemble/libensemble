import ipaddress
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import (BaseConfig, BaseModel, Field, root_validator,
                      validator)

from libensemble.utils.type_checkers import (
    _check_any_workers_and_disable_rm_if_tcp, _check_exit_criteria, _check_H0,
    _check_output_fields, _MPICommValidationModel)

BaseConfig.arbitrary_types_allowed = True

__all__ = ["SimSpecs", "GenSpecs", "AllocSpecs", "ExitCriteria", "LibeSpecs", "EnsembleSpecs"]


class SimSpecs(BaseModel):
    sim_f: Callable
    inputs: List[str] = Field([], alias="in")
    persis_in: Optional[List[str]]
    # list of tuples for dtype construction
    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]] = []
    funcx_endpoint: Optional[str] = ""
    user: Optional[Dict]
    _out_dtype: np.dtype = []

    @root_validator
    def set_out_dtype(cls, values):
        if values.get("out"):
            values["_out_dtype"] = np.dtype(values.get("out"))
        return values


class GenSpecs(BaseModel):
    gen_f: Callable
    inputs: Optional[List[str]] = Field([], alias="in")
    persis_in: Optional[List[str]]
    # list of tuples for dtype construction
    out: List[Union[Tuple[str, Any], Tuple[str, Any, Union[int, Tuple]]]] = []
    funcx_endpoint: Optional[str] = ""
    user: Optional[Dict]
    _out_dtype: np.dtype = []

    @root_validator
    def set_out_dtype(cls, values):
        if values.get("out"):
            values["_out_dtype"] = np.dtype(values.get("out"))
        return values


class AllocSpecs(BaseModel):
    alloc_f: Callable
    user: Optional[Dict]


class ExitCriteria(BaseModel):
    sim_max: Optional[int] = 100
    gen_max: Optional[int]
    wallclock_max: Optional[float]
    stop_val: Optional[Tuple[str, float]]


class LibeSpecs(BaseModel):
    abort_on_exception: Optional[bool] = True
    enforce_worker_core_bounds: Optional[bool] = False
    authkey: Optional[str] = f"libE_auth_{random.randrange(99999)}"
    disable_resource_manager: Optional[bool] = False
    dedicated_mode: Optional[bool] = False
    comms: str = "mpi"
    resource_info: Optional[Dict] = {}
    disable_log_files: Optional[bool] = False
    final_fields: Optional[List[str]] = []
    ip: Optional[ipaddress.IPv4Address] = None
    kill_canceled_sims: Optional[bool] = True
    mpi_comm: Optional[_MPICommValidationModel] = None  # see utils/type_checkers.py
    num_resource_sets: Optional[int]
    nworkers: Optional[int]
    port: Optional[int] = 0
    profile: Optional[bool] = False
    safe_mode: Optional[bool] = True
    save_every_k_gens: Optional[int] = 0
    save_every_k_sims: Optional[int] = 0
    save_H_and_persis_on_abort: Optional[bool] = True
    scheduler_opts: Optional[Dict] = {}
    stats_fmt: Optional[Dict] = {}
    use_persis_return_gen: Optional[bool] = False
    use_persis_return_sim: Optional[bool] = False
    workerID: Optional[int]
    worker_timeout: Optional[int] = 1
    zero_resource_workers: Optional[List[int]] = []
    worker_cmd: Optional[List[str]]
    workers: Optional[List[str]]
    ensemble_copy_back: Optional[bool] = False
    ensemble_dir_path: Optional[str] = "./ensemble"
    use_worker_dirs: Optional[bool] = False
    sim_dirs_make: Optional[bool] = False
    sim_dir_copy_files: Optional[List[str]] = []
    sim_dir_symlink_files: Optional[List[str]] = []
    sim_input_dir: Optional[str] = ""
    gen_dirs_make: Optional[bool] = False
    gen_dir_copy_files: Optional[List[str]] = []
    gen_dir_symlink_files: Optional[List[str]] = []
    gen_input_dir: Optional[str] = ""

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
        return _check_any_workers_and_disable_rm_if_tcp(cls, values)


class EnsembleSpecs(BaseModel):
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
        return _check_exit_criteria(cls, values)

    @root_validator
    def check_output_fields(cls, values):
        return _check_output_fields(cls, values)

    @root_validator
    def set_ensemble_nworkers(cls, values):
        if values.get("libE_specs"):
            values["nworkers"] = values["libE_specs"].nworkers
            return values

    @root_validator
    def check_H0(cls, values):
        return _check_H0(cls, values)
