"""
Save some space in specs.py by moving some validation functions to here.
Reference the models in that file.
"""

import logging
import secrets
from pathlib import Path

import numpy as np

from libensemble.tools.fields_keys import libE_fields
from libensemble.utils.misc import specs_checker_getattr as scg
from libensemble.utils.misc import specs_checker_setattr as scs

logger = logging.getLogger(__name__)


def _check_exit_criteria(values):
    if scg(values, "exit_criteria").stop_val is not None:
        stop_name = scg(values, "exit_criteria").stop_val[0]
        sim_out_names = [e[0] for e in scg(values, "sim_specs").outputs]
        gen_out_names = [e[0] for e in scg(values, "gen_specs").outputs]
        assert stop_name in sim_out_names + gen_out_names, f"Can't stop on {stop_name} if it's not in a sim/gen output"
    return values


def _check_set_gen_specs_from_variables(values):
    if not len(scg(values, "outputs")):
        if scg(values, "generator") and len(scg(values, "generator").gen_specs["out"]):
            scs(values, "outputs", scg(values, "generator").gen_specs["out"])
    return values


def _check_H0(values):
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


def _check_any_workers_and_disable_rm_if_tcp(values):
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


def _check_set_workflow_dir(values):
    if scg(values, "use_workflow_dir") and len(str(scg(values, "workflow_dir_path"))) <= 1:
        scs(values, "workflow_dir_path", Path("./workflow_" + secrets.token_hex(3)).absolute())
    elif len(str(scg(values, "workflow_dir_path"))) > 1:
        if not scg(values, "use_workflow_dir"):
            scs(values, "use_workflow_dir", True)
        scs(values, "workflow_dir_path", Path(scg(values, "workflow_dir_path")).absolute())
    return values


def _check_set_calc_dirs_on_input_dir(values):
    if scg(values, "sim_input_dir") and not scg(values, "sim_dirs_make"):
        scs(values, "sim_dirs_make", True)
    if scg(values, "gen_input_dir") and not scg(values, "gen_dirs_make"):
        scs(values, "gen_dirs_make", True)
    return values


def _check_logical_cores(values):
    if scg(values, "cores_per_node") and scg(values, "logical_cores_per_node"):
        assert (
            scg(values, "logical_cores_per_node") % scg(values, "cores_per_node") == 0
        ), "Logical cores doesn't divide evenly into cores"
    return values
