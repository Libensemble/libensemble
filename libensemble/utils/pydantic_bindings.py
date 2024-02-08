from pydantic import Field, create_model

from libensemble import specs
from libensemble.resources import platforms
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
    check_provided_ufuncs,
    check_valid_comms_type,
    check_valid_in,
    check_valid_out,
    enable_save_H_when_every_K,
    genf_set_in_out_from_attrs,
    set_platform_specs_to_class,
    set_workflow_dir,
    simf_set_in_out_from_attrs,
)

if pydanticV1:
    from pydantic import BaseConfig
    from pydantic import validate_arguments as libE_wrapper  # noqa: F401

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

    specs.LibeSpecs.Config = Config
    specs._EnsembleSpecs.Config = Config

elif pydanticV2:
    from pydantic import ConfigDict
    from pydantic import validate_call as libE_wrapper  # noqa: F401
    from pydantic.fields import FieldInfo

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    specs.SimSpecs.model_config = model_config
    specs.GenSpecs.model_config = model_config
    specs.AllocSpecs.model_config = model_config
    specs.LibeSpecs.model_config = model_config
    specs.ExitCriteria.model_config = model_config
    specs._EnsembleSpecs.model_config = model_config
    platforms.Platform.model_config = model_config

    specs.SimSpecs.model_fields["inputs"] = FieldInfo.merge_field_infos(
        specs.SimSpecs.model_fields["inputs"], Field(alias="in")
    )

    specs.SimSpecs.model_fields["outputs"] = FieldInfo.merge_field_infos(
        specs.SimSpecs.model_fields["outputs"], Field(alias="out")
    )

    specs.GenSpecs.model_fields["inputs"] = FieldInfo.merge_field_infos(
        specs.GenSpecs.model_fields["inputs"], Field(alias="in")
    )

    specs.GenSpecs.model_fields["outputs"] = FieldInfo.merge_field_infos(
        specs.GenSpecs.model_fields["outputs"], Field(alias="out")
    )

    specs.AllocSpecs.model_fields["outputs"] = FieldInfo.merge_field_infos(
        specs.AllocSpecs.model_fields["outputs"], Field(alias="out")
    )

    specs.SimSpecs.model_rebuild(force=True)
    specs.GenSpecs.model_rebuild(force=True)
    specs.AllocSpecs.model_rebuild(force=True)
    specs.LibeSpecs.model_rebuild(force=True)
    specs.ExitCriteria.model_rebuild(force=True)
    specs._EnsembleSpecs.model_rebuild(force=True)
    platforms.Platform.model_rebuild(force=True)

specs.SimSpecs = create_model(
    "SimSpecs",
    __base__=specs.SimSpecs,
    __validators__={
        "check_valid_out": check_valid_out,
        "check_valid_in": check_valid_in,
        "simf_set_in_out_from_attrs": simf_set_in_out_from_attrs,
    },
)

specs.GenSpecs = create_model(
    "GenSpecs",
    __base__=specs.GenSpecs,
    __validators__={
        "check_valid_out": check_valid_out,
        "check_valid_in": check_valid_in,
        "genf_set_in_out_from_attrs": genf_set_in_out_from_attrs,
    },
)

specs.LibeSpecs = create_model(
    "LibeSpecs",
    __base__=specs.LibeSpecs,
    __validators__={
        "check_valid_comms_type": check_valid_comms_type,
        "set_platform_specs_to_class": set_platform_specs_to_class,
        "check_input_dir_exists": check_input_dir_exists,
        "check_inputs_exist": check_inputs_exist,
        "check_any_workers_and_disable_rm_if_tcp": check_any_workers_and_disable_rm_if_tcp,
        "enable_save_H_when_every_K": enable_save_H_when_every_K,
        "set_workflow_dir": set_workflow_dir,
    },
)

specs._EnsembleSpecs = create_model(
    "_EnsembleSpecs",
    __base__=specs._EnsembleSpecs,
    __validators__={
        "check_exit_criteria": check_exit_criteria,
        "check_output_fields": check_output_fields,
        "check_H0": check_H0,
        "check_provided_ufuncs": check_provided_ufuncs,
    },
)

platforms.Platform = create_model(
    "Platform",
    __base__=platforms.Platform,
    __validators__={
        "check_gpu_setting_type": check_gpu_setting_type,
        "check_mpi_runner_type": check_mpi_runner_type,
        "check_logical_cores": check_logical_cores,
    },
)
