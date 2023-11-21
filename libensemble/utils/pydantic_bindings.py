from libensemble.resources.platforms import Platform
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs, _EnsembleSpecs
from libensemble.utils.misc import pydanticV1, pydanticV2
from libensemble.utils.validators import (
    _UFUNC_INVALID_ERR,
    _UNRECOGNIZED_ERR,
    check_any_workers_and_disable_rm_if_tcp,
    check_exit_criteria,
    check_gpu_setting_type,
    check_H0,
    check_input_dir_exists,
    check_inputs_exist,
    check_logical_cores,
    check_mpi_runner_type,
    check_output_fields,
    check_valid_comms_type,
    check_valid_in,
    check_valid_out,
    enable_save_H_when_every_K,
    set_ensemble_nworkers,
    set_platform_specs_to_class,
    set_workflow_dir,
)

if pydanticV1:
    from pydantic import BaseConfig, Field

    BaseConfig.arbitrary_types_allowed = True
    BaseConfig.allow_population_by_field_name = True
    BaseConfig.extra = "allow"
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
    from pydantic import ConfigDict, Field

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="allow", validate_assignment=True
    )

    SimSpecs.model_config = model_config
    GenSpecs.model_config = model_config
    LibeSpecs.model_config = model_config
    ExitCriteria.model_config = model_config
    _EnsembleSpecs.model_config = model_config
    Platform.model_config = model_config

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

Platform.check_gpu_setting_type = check_gpu_setting_type
Platform.check_mpi_runner_type = check_mpi_runner_type
Platform.check_logical_cores = check_logical_cores

SimSpecs.inputs = Field(default=[], serialization_alias="in")
SimSpecs.outputs = Field(default=[], serialization_alias="out")
GenSpecs.inputs = Field(default=[], serialization_alias="in")
GenSpecs.outputs = Field(default=[], serialization_alias="out")
AllocSpecs.outputs = Field(default=[], serialization_alias="out")
