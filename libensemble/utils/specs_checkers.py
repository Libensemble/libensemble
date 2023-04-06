"""
Save some space in specs.py by moving some validation functions to here.
Reference the models in that file.
"""

import logging

import numpy as np

from libensemble.tools.fields_keys import libE_fields

logger = logging.getLogger(__name__)


def _check_exit_criteria(values: dict) -> dict:
    if "stop_val" in values.get("exit_criteria"):
        stop_name = values.get("exit_criteria").stop_val[0]
        sim_out_names = [e[0] for e in values.get("sim_specs").out]
        gen_out_names = [e[0] for e in values.get("gen_specs").out]
        assert stop_name in sim_out_names + gen_out_names, f"Can't stop on {stop_name} if it's not in a sim/gen output"
    return values


def _check_output_fields(values: dict) -> dict:
    out_names = [e[0] for e in libE_fields]
    if values.get("H0") is not None and values.get("H0").dtype.names is not None:
        out_names += list(values.get("H0").dtype.names)
    out_names += [e[0] for e in values.get("sim_specs").out]
    if values.get("gen_specs"):
        out_names += [e[0] for e in values.get("gen_specs").out]
    if values.get("alloc_specs"):
        out_names += [e[0] for e in values.get("alloc_specs").out]

    if values.get("libE_specs"):
        for name in values.get("libE_specs").final_fields:
            assert name in out_names, (
                name + " in libE_specs['fields_keys'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )

    for name in values.get("sim_specs").inputs:
        assert name in out_names, (
            name + " in sim_specs['in'] is not in sim_specs['out'], "
            "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
        )

    if values.get("gen_specs"):
        for name in values.get("gen_specs").inputs:
            assert name in out_names, (
                name + " in gen_specs['in'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )
    return values


def _check_H0(values: dict) -> dict:
    if values.get("H0").size > 0:
        H0 = values.get("H0")
        specs = [values.get("sim_specs"), values.get("gen_specs")]
        specs_dtype_list = list(set(libE_fields + sum([k.out or [] for k in specs if k], [])))
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


def _check_any_workers_and_disable_rm_if_tcp(values: dict) -> dict:
    comms_type = values.get("comms")
    if comms_type in ["local", "tcp"]:
        if values.get("nworkers"):
            assert values.get("nworkers") >= 1, "Must specify at least one worker"
        else:
            if comms_type == "tcp":
                assert values.get("workers"), "Without nworkers, must specify worker hosts on TCP"
    if comms_type == "tcp":
        values["disable_resource_manager"] = True  # Resource management not supported with TCP
    return values


class MPI_Communicator:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    def validate(cls, comm: "MPIComm") -> "MPIComm":  # noqa: F821
        from mpi4py import MPI

        if comm == MPI.COMM_NULL:
            logger.manager_warning("*WARNING* libEnsemble detected a NULL communicator")
        else:
            assert comm.Get_size() > 1, "Manager only - must be at least one worker (2 MPI tasks)"
        return comm
