import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import field_validator, model_validator

from libensemble.resources.platforms import Platform
from libensemble.utils.specs_checkers import (
    _check_any_workers_and_disable_rm_if_tcp,
    _check_exit_criteria,
    _check_H0,
    _check_logical_cores,
    _check_set_calc_dirs_on_input_dir,
    _check_set_gen_specs_from_variables,
    _check_set_workflow_dir,
)

_UNRECOGNIZED_ERR = "Unrecognized field. Check closely for typos, or libEnsemble's docs"
_UFUNC_INVALID_ERR = "Specified sim_f or gen_f is not callable. It should be a user function"
_OUT_DTYPE_ERR = "Unable to coerce into a NumPy dtype. It should be a list of 2-tuples or 3-tuples"
_IN_INVALID_ERR = "Value should be a list of field names (a list of strings)"


def detect_comms_env():
    """Return local or MPI comms based on env variables"""
    mpi_vars = ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE"]
    comms_type = "local"
    for var in mpi_vars:
        value = os.getenv(var)
        if value is not None:
            if int(value) > 1:
                comms_type = "mpi"
                break
    return comms_type


def default_comms(values):
    if "comms" not in values:
        if values.get("nworkers") is not None:
            values["comms"] = detect_comms_env()
        else:
            values["comms"] = "mpi"
    return values


def check_valid_out(cls, v):
    try:
        _ = np.dtype(v)
    except TypeError:
        raise ValueError(_OUT_DTYPE_ERR.format(v))
    else:
        return v


def check_valid_in(cls, v):
    if not all(isinstance(s, str) for s in v):
        raise ValueError(_IN_INVALID_ERR)
    return v


def check_valid_comms_type(cls, value):
    assert value in ["mpi", "local", "threads", "tcp"], "Invalid comms type"
    return value


def set_platform_specs_to_class(cls, value) -> Platform:
    if isinstance(value, dict):
        value = Platform(**value)
    return value


def check_input_dir_exists(cls, value):
    if value:
        if isinstance(value, str):
            value = Path(value).absolute()
        assert value.exists(), "value does not refer to an existing path"
        assert value != Path("."), "Value can't refer to the current directory ('.' or Path('.'))."
    return value


def check_inputs_exist(cls, value):
    value = [Path(path).absolute() for path in value]
    for f in value:
        assert f.exists(), f"'{f}' in Value does not refer to an existing path."
    return value


def check_gpu_setting_type(cls, value):
    if value is not None:
        assert value in [
            "runner_default",
            "env",
            "option_gpus_per_node",
            "option_gpus_per_task",
        ], "Invalid label for GPU specification type"
    return value


def check_mpi_runner_type(cls, value):
    if value is not None:
        assert value in ["mpich", "openmpi", "aprun", "srun", "jsrun", "msmpi", "custom"], "Invalid MPI runner name"
    return value


# SPECS VALIDATORS #####

check_valid_out = field_validator("outputs")(classmethod(check_valid_out))
check_valid_in = field_validator("inputs", "persis_in")(classmethod(check_valid_in))
check_valid_comms_type = field_validator("comms")(classmethod(check_valid_comms_type))
set_platform_specs_to_class = field_validator("platform_specs")(classmethod(set_platform_specs_to_class))
check_input_dir_exists = field_validator("sim_input_dir", "gen_input_dir")(classmethod(check_input_dir_exists))
check_inputs_exist = field_validator(
    "sim_dir_copy_files", "sim_dir_symlink_files", "gen_dir_copy_files", "gen_dir_symlink_files"
)(classmethod(check_inputs_exist))
check_gpu_setting_type = field_validator("gpu_setting_type")(classmethod(check_gpu_setting_type))
check_mpi_runner_type = field_validator("mpi_runner")(classmethod(check_mpi_runner_type))


@model_validator(mode="after")
def check_any_workers_and_disable_rm_if_tcp(self):
    return _check_any_workers_and_disable_rm_if_tcp(self)


@model_validator(mode="before")
def set_default_comms(cls, values):
    return default_comms(values)


@model_validator(mode="after")
def enable_save_H_when_every_K(self):
    if not self.__dict__.get("save_H_on_completion") and (
        self.__dict__.get("save_every_k_sims", 0) > 0 or self.__dict__.get("save_every_k_gens", 0) > 0
    ):
        self.__dict__["save_H_on_completion"] = True
    return self


@model_validator(mode="after")
def set_workflow_dir(self):
    return _check_set_workflow_dir(self)


@model_validator(mode="after")
def set_calc_dirs_on_input_dir(self):
    return _check_set_calc_dirs_on_input_dir(self)


@model_validator(mode="after")
def check_exit_criteria(self):
    return _check_exit_criteria(self)


@model_validator(mode="after")
def check_H0(self):
    return _check_H0(self)


@model_validator(mode="after")
def check_set_gen_specs_from_variables(self):
    return _check_set_gen_specs_from_variables(self)


@model_validator(mode="after")
def check_provided_ufuncs(self):
    assert hasattr(self.sim_specs, "sim_f"), "Simulation function not provided to SimSpecs."
    assert isinstance(self.sim_specs.sim_f, Callable), "Simulation function is not callable."

    if self.alloc_specs.alloc_f.__name__ != "give_pregenerated_sim_work":
        assert hasattr(self.gen_specs, "gen_f"), "Generator function not provided to GenSpecs."
        assert (
            isinstance(self.gen_specs.gen_f, Callable) if self.gen_specs.gen_f is not None else True
        ), "Generator function is not callable."

    return self


@model_validator(mode="after")
def simf_set_in_out_from_attrs(self):
    if hasattr(self.__dict__.get("sim_f"), "inputs") and not self.__dict__.get("inputs"):
        self.__dict__["inputs"] = self.__dict__.get("sim_f").inputs
    if hasattr(self.__dict__.get("sim_f"), "outputs") and not self.__dict__.get("outputs"):
        self.__dict__["outputs"] = self.__dict__.get("sim_f").outputs
    if hasattr(self.__dict__.get("sim_f"), "persis_in") and not self.__dict__.get("persis_in"):
        self.__dict__["persis_in"] = self.__dict__.get("sim_f").persis_in
    return self


@model_validator(mode="after")
def genf_set_in_out_from_attrs(self):
    if hasattr(self.__dict__.get("gen_f"), "inputs") and not self.__dict__.get("inputs"):
        self.__dict__["inputs"] = self.__dict__.get("gen_f").inputs
    if hasattr(self.__dict__.get("gen_f"), "outputs") and not self.__dict__.get("outputs"):
        self.__dict__["outputs"] = self.__dict__.get("gen_f").outputs
    if hasattr(self.__dict__.get("gen_f"), "persis_in") and not self.__dict__.get("persis_in"):
        self.__dict__["persis_in"] = self.__dict__.get("gen_f").persis_in
    return self


# RESOURCES VALIDATORS #####


@model_validator(mode="after")
def check_logical_cores(self):
    return _check_logical_cores(self)
