import pydantic

from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs, _EnsembleSpecs
from libensemble.utils.validators import (
    _UFUNC_INVALID_ERR,
    _UNRECOGNIZED_ERR,
    check_any_workers_and_disable_rm_if_tcp,
    check_exit_criteria,
    check_H0,
    check_input_dir_exists,
    check_inputs_exist,
    check_output_fields,
    check_valid_comms_type,
    check_valid_in,
    check_valid_out,
    enable_save_H_when_every_K,
    set_ensemble_nworkers,
    set_platform_specs_to_class,
    set_workflow_dir,
)

SimSpecs.check_valid_out = check_valid_out
SimSpecs.check_valid_in = check_valid_in
GenSpecs.check_valid_out = check_valid_out
GenSpecs.check_valid_in = check_valid_in
LibeSpecs.check_valid_comms_type = check_valid_comms_type
LibeSpecs.set_platform_specs_to_class = set_platform_specs_to_class
LibeSpecs.check_input_dir_exists = check_input_dir_exists
LibeSpecs.check_inputs_exist = check_inputs_exist
LibeSpecs.check_any_workers_and_disable_rm_if_tcp = check_any_workers_and_disable_rm_if_tcp
LibeSpecs.enable_save_H_when_every_K = enable_save_H_when_every_K
LibeSpecs.set_workflow_dir = set_workflow_dir
_EnsembleSpecs.check_exit_criteria = check_exit_criteria
_EnsembleSpecs.check_output_fields = check_output_fields
_EnsembleSpecs.set_ensemble_nworkers = set_ensemble_nworkers
_EnsembleSpecs.check_H0 = check_H0

pydantic_version = pydantic.__version__[0]

pydanticV1 = pydantic_version == "1"
pydanticV2 = pydantic_version == "2"

if pydanticV1:
    from pydantic import BaseConfig

    BaseConfig.arbitrary_types_allowed = True
    BaseConfig.allow_population_by_field_name = True
    BaseConfig.extra = "forbid"
    BaseConfig.error_msg_templates = {
        "value_error.extra": _UNRECOGNIZED_ERR,
        "type_error.callable": _UFUNC_INVALID_ERR,
    }
    BaseConfig.validate_assignment = True

    class Config:
        arbitrary_types_allowed = True

    LibeSpecs.Config = Config
    _EnsembleSpecs.Config = Config


elif pydanticV2:
    from pydantic import ConfigDict

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    SimSpecs.model_config = model_config
    GenSpecs.model_config = model_config
    LibeSpecs.model_config = model_config
    ExitCriteria.model_config = model_config
    _EnsembleSpecs.model_config = model_config


def specs_dump(specs, **kwargs):
    if pydanticV1:
        return specs.dict(**kwargs)
    elif pydanticV2:
        return specs.model_dump(**kwargs)


def specs_checker_getattr(obj, key):
    if pydanticV1:  # dict
        return obj.get(key)
    elif pydanticV2:  # actual obj
        return getattr(obj, key)


def specs_check_setattr(obj, key, value):
    if pydanticV1:  # dict
        obj[key] = value
    elif pydanticV2:  # actual obj
        obj.__dict__[key] = value
