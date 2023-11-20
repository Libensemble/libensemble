import secrets
from pathlib import Path
from typing import Union

import numpy as np

from libensemble.resources.platforms import Platform
from libensemble.utils.pydantic_bindings import pydanticV1, pydanticV2
from libensemble.utils.specs_checkers import (
    _check_any_workers_and_disable_rm_if_tcp,
    _check_exit_criteria,
    _check_H0,
    _check_output_fields,
)

_UNRECOGNIZED_ERR = "Unrecognized field. Check closely for typos, or libEnsemble's docs"
_UFUNC_INVALID_ERR = "Specified sim_f or gen_f is not callable. It should be a user function"
_OUT_DTYPE_ERR = "unable to coerce into a NumPy dtype. It should be a list of 2-tuples or 3-tuples"
_IN_INVALID_ERR = "value should be a list of field names (a list of strings)"

""" Pydantic validation logic, implemented in both V1 and V2"""

if pydanticV1:
    from pydantic import root_validator, validator

    @validator("outputs", pre=True)
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

    @validator("comms")
    def check_valid_comms_type(cls, value):
        assert value in ["mpi", "local", "local_threading", "tcp"], "Invalid comms type"
        return value

    @validator("platform_specs")
    def set_platform_specs_to_class(cls, value) -> Platform:
        if isinstance(value, dict):
            value = Platform(**value)
        return value

    @validator("sim_input_dir", "gen_input_dir")
    def check_input_dir_exists(cls, value):
        if value:
            if isinstance(value, str):
                value = Path(value).absolute()
            assert value.exists(), "value does not refer to an existing path"
            assert value != Path("."), "Value can't refer to the current directory ('.' or Path('.'))."
        return value

    @validator("sim_dir_copy_files", "sim_dir_symlink_files", "gen_dir_copy_files", "gen_dir_symlink_files")
    def check_inputs_exist(cls, value):
        value = [Path(path).absolute() for path in value]
        for f in value:
            assert f.exists(), f"'{f}' in Value does not refer to an existing path."
        return value

    @root_validator
    def check_any_workers_and_disable_rm_if_tcp(cls, values):
        return _check_any_workers_and_disable_rm_if_tcp(values)

    @root_validator(pre=True)
    def enable_save_H_when_every_K(cls, values):
        if "save_H_on_completion" not in values and (
            values.get("save_every_k_sims", 0) > 0 or values.get("save_every_k_gens", 0) > 0
        ):
            values["save_H_on_completion"] = True
        return values

    @root_validator
    def set_workflow_dir(cls, values):
        if values.get("use_workflow_dir") and len(str(values.get("workflow_dir_path"))) <= 1:
            values["workflow_dir_path"] = Path(
                "./workflow_" + secrets.token_hex(3)
            ).absolute()  # should avoid side-effects. make dir later
        elif len(str(values.get("workflow_dir_path"))) > 1:
            if not values.get("use_workflow_dir"):
                values["use_workflow_dir"] = True
            values["workflow_dir_path"] = Path(values["workflow_dir_path"]).absolute()
        return values

    @root_validator
    def check_exit_criteria(cls, values):
        return _check_exit_criteria(values)

    @root_validator
    def check_output_fields(cls, values):
        return _check_output_fields(values)

    @root_validator
    def set_ensemble_nworkers(cls, values):
        if values.get("libE_specs"):
            values["nworkers"] = values["libE_specs"].nworkers
        return values

    @root_validator
    def check_H0(cls, values):
        if values.get("H0") is not None:
            return _check_H0(values)
        return values

elif pydanticV2:
    from pydantic import field_validator, model_validator

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
        if "use_workflow_dir" in self.__dict__ and len(str(self.__dict__.get("workflow_dir_path"))) <= 1:
            self.__dict__["workflow_dir_path"] = Path(
                "./workflow_" + secrets.token_hex(3)
            ).absolute()  # should avoid side-effects. make dir later
        elif len(str(self.workflow_dir_path)) > 1:
            if not self.use_workflow_dir:
                self.__dict__["use_workflow_dir"] = True
            self.__dict__["workflow_dir_path"] = Path(self.workflow_dir_path).absolute()
        return self

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
