import logging
import os
import secrets
from collections.abc import Callable
from pathlib import Path

import numpy as np

from libensemble.resources.platforms import Platform
from libensemble.tools.fields_keys import libE_fields
from libensemble.utils.misc import specs_checker_getattr as scg
from libensemble.utils.misc import specs_checker_setattr as scs

logger = logging.getLogger(__name__)

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


def check_any_workers_and_disable_rm_if_tcp(values):
    comms_type = scg(values, "comms")
    if comms_type in ["local", "tcp"]:
        if scg(values, "nworkers"):
            assert scg(values, "nworkers") >= 1, "Must specify at least one worker"
        else:
            if comms_type == "tcp":
                assert scg(values, "workers"), "Without nworkers, must specify worker hosts on TCP"
    if comms_type == "tcp":
        scs(values, "disable_resource_manager", True)  # Resource management not supported with TCP
    return values


def set_default_comms(cls, values):
    return default_comms(values)


def enable_save_H_when_every_K(self):
    if not self.__dict__.get("save_H_on_completion") and (
        self.__dict__.get("save_every_k_sims", 0) > 0 or self.__dict__.get("save_every_k_gens", 0) > 0
    ):
        self.__dict__["save_H_on_completion"] = True
    return self


def set_workflow_dir(values):
    if scg(values, "use_workflow_dir") and len(str(scg(values, "workflow_dir_path"))) <= 1:
        scs(values, "workflow_dir_path", Path("./workflow_" + secrets.token_hex(3)).absolute())
    elif len(str(scg(values, "workflow_dir_path"))) > 1:
        if not scg(values, "use_workflow_dir"):
            scs(values, "use_workflow_dir", True)
        scs(values, "workflow_dir_path", Path(scg(values, "workflow_dir_path")).absolute())
    return values


def set_calc_dirs_on_input_dir(values):
    if scg(values, "sim_input_dir") and not scg(values, "sim_dirs_make"):
        scs(values, "sim_dirs_make", True)
    if scg(values, "gen_input_dir") and not scg(values, "gen_dirs_make"):
        scs(values, "gen_dirs_make", True)
    return values


def check_exit_criteria(values):
    if scg(values, "exit_criteria").stop_val is not None:
        stop_name = scg(values, "exit_criteria").stop_val[0]
        sim_out_names = [e[0] for e in scg(values, "sim_specs").outputs]
        gen_out_names = [e[0] for e in scg(values, "gen_specs").outputs]
        assert stop_name in sim_out_names + gen_out_names, f"Can't stop on {stop_name} if it's not in a sim/gen output"
    return values


def check_H0(values):
    if scg(values, "H0").size > 0:
        H0 = scg(values, "H0")
        specs = [scg(values, "sim_specs"), scg(values, "gen_specs")]
        specs_dtype_list = list(set(libE_fields + sum([k.outputs or [] for k in specs if k], [])))
        specs_dtype_fields = [i[0] for i in specs_dtype_list]
        specs_inputs_list = list(set(sum([k.inputs + k.persis_in or [] for k in specs if k], [])))
        Dummy_H = np.zeros(1 + len(H0), dtype=specs_dtype_list)

        # should check that new fields compatible with sim/gen specs, if any?

        for field in specs_inputs_list:
            assert field in list(H0.dtype.names) + specs_dtype_fields, f"{field} not in H0 although expected as input"

        assert "sim_ended" not in H0.dtype.names or np.all(
            H0["sim_started"] == H0["sim_ended"]
        ), "H0 contains unreturned or invalid points"

        def _check_consistent_field(name, field0, field1):
            """Checks that new field (field1) is compatible with an old field (field0)."""
            assert field0.ndim == field1.ndim, f"H0 and H have different ndim for field {name}"
            assert np.all(
                np.array(field1.shape) >= np.array(field0.shape)
            ), f"H too small to receive all components of H0 in field {name}"

        for field in H0.dtype.names:
            if field in specs_dtype_list:
                _check_consistent_field(field, H0[field], Dummy_H[field])
    return values


def check_set_gen_specs_from_variables(values):
    if not len(scg(values, "outputs")):
        generator = scg(values, "generator")
        if generator and hasattr(generator, "gen_specs"):
            out = generator.gen_specs.get("out", [])
            if len(out):
                scs(values, "outputs", out)
    return values


def check_provided_ufuncs(self):
    assert hasattr(self.sim_specs, "sim_f"), "Simulation function not provided to SimSpecs."
    assert isinstance(self.sim_specs.sim_f, Callable), "Simulation function is not callable."

    if self.alloc_specs.alloc_f.__name__ != "give_pregenerated_sim_work":
        assert hasattr(self.gen_specs, "gen_f"), "Generator function not provided to GenSpecs."
        assert (
            isinstance(self.gen_specs.gen_f, Callable) if self.gen_specs.gen_f is not None else True
        ), "Generator function is not callable."

    return self


def check_logical_cores(values):
    if scg(values, "cores_per_node") and scg(values, "logical_cores_per_node"):
        assert (
            scg(values, "logical_cores_per_node") % scg(values, "cores_per_node") == 0
        ), "Logical cores doesn't divide evenly into cores"
    return values
