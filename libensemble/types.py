import random
import typing
import ipaddress
from pathlib import Path
from typing import Dict, Callable, List, Any, Tuple, Union, Optional

import numpy as np
from pydantic import BaseModel, validator, BaseSettings, PyObject
from libensemble.tools.fields_keys import allowed_libE_spec_keys, libE_fields


class SimSpecs(BaseModel):
    function: Callable
    inputs: List
    outputs: Tuple[str, Any, Union[int, Tuple]]
    funcx_endpoint: Optional[str]
    user: Optional[Dict]


class GenSpecs(BaseModel):
    function: Callable
    inputs: Optional[List]
    outputs: Tuple[str, Any, Union[int, Tuple]]
    funcx_endpoint: Optional[str]
    user: Optional[Dict]


class AllocSpecs(BaseModel):
    function: Callable
    user: Optional[Dict]


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


class LibeSpecs(BaseSettings):
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


class Ensemble(BaseModel):
    H0: np.ndarray
    libE_specs: LibeSpecs
    persis_info: Dict
    sim_specs: SimSpecs
    gen_specs: GenSpecs

    class Config:
        arbitrary_types_allowed = True
