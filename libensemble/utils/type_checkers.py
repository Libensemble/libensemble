"""
Save some space in types.py by moving some validation functions to here.
Reference the models in that file.
"""

import numpy as np
from libensemble.tools.fields_keys import libE_fields


def _check_exit_criteria(cls, values):
    if "stop_val" in values.get("exit_criteria"):
        stop_name = values.get("exit_criteria").stop_val[0]
        sim_out_names = [e[0] for e in values.get("sim_specs").out]
        gen_out_names = [e[0] for e in values.get("gen_specs").out]
        assert stop_name in sim_out_names + gen_out_names, f"Can't stop on {stop_name} if it's not in a sim/gen output"
    return values


def _check_output_fields(cls, values):
    out_names = [e[0] for e in libE_fields]
    if values.get("H0") is not None and values.get("H0").dtype.names is not None:
        out_names += list(values.get("H0").dtype.names)
    out_names += [e[0] for e in values.get("sim_specs").out]
    if values.get("gen_specs"):
        out_names += [e[0] for e in values.get("gen_specs").out]

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


def _check_H0(cls, values):
    if values.get("H0").size > 0:
        H0 = values.get("H0")
        specs = [values.get("sim_specs"), values.get("alloc_specs"), values.get("gen_specs")]
        dtype_list = list(set(libE_fields + sum([k.out or [] for k in specs if k], [])))
        Dummy_H = np.zeros(1 + len(H0), dtype=dtype_list)

        fields = H0.dtype.names

        assert set(fields).issubset(
            set(Dummy_H.dtype.names)
        ), f"H0 contains fields {set(fields).difference(set(Dummy_H.dtype.names))} not in the History."
        assert "sim_ended" not in fields or np.all(
            H0["sim_started"] == H0["sim_ended"]
        ), "H0 contains unreturned or invalid points"

        def _check_consistent_field(name, field0, field1):
            """Checks that new field (field1) is compatible with an old field (field0)."""
            assert field0.ndim == field1.ndim, f"H0 and H have different ndim for field {name}"
            assert np.all(
                np.array(field1.shape) >= np.array(field0.shape)
            ), f"H too small to receive all components of H0 in field {name}"

        for field in fields:
            _check_consistent_field(field, H0[field], Dummy_H[field])
    return values


def _check_any_workers_and_disable_rm_if_tcp(cls, values):
    comms_type = values.get("comms")
    if comms_type in ["local", "tcp"]:
        if values.get("nworkers"):
            assert values.get("nworkers") >= 1, "Must specify at least one worker"
        else:
            assert values.get("workers"), "Without nworkers, must specify worker hosts on TCP"
    if comms_type == "tcp":
        values["disable_resource_manager"] = True  # Resource management not supported with TCP
    return values


class _MPICommValidationModel:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    def validate(cls, comm):
        assert comm.Get_size() > 1, "Manager only - must be at least one worker (2 MPI tasks)"
        return comm
