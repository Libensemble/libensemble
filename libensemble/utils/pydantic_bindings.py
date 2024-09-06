import sys

from libensemble import specs
from libensemble.resources import platforms
from libensemble.utils.pydantic_modules import BaseConfig, create_model
from libensemble.utils.pydantic_modules import validate_arguments as libE_wrapper  # noqa: F401
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
    set_calc_dirs_on_input_dir,
    set_default_comms,
    set_platform_specs_to_class,
    set_workflow_dir,
    simf_set_in_out_from_attrs,
)

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

# the create_model function removes fields for rendering in docs
if "sphinx" not in sys.modules:

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
            "set_default_comms": set_default_comms,
            "set_workflow_dir": set_workflow_dir,
            "set_calc_dirs_on_input_dir": set_calc_dirs_on_input_dir,
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
