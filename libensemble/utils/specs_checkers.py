"""
Save some space in specs.py by moving some validation functions to here.
Reference the models in that file.
"""

import logging

import numpy as np

from libensemble.tools.fields_keys import libE_fields

logger = logging.getLogger(__name__)


def _check_exit_criteria(EnsembleSpecs):
    if "stop_val" in EnsembleSpecs.exit_criteria:
        stop_name = EnsembleSpecs.exit_criteria.stop_val[0]
        sim_out_names = [e[0] for e in EnsembleSpecs.sim_specs.outputs]
        gen_out_names = [e[0] for e in EnsembleSpecs.gen_specs.outputs]
        assert stop_name in sim_out_names + gen_out_names, f"Can't stop on {stop_name} if it's not in a sim/gen output"
    return EnsembleSpecs


def _check_output_fields(EnsembleSpecs):
    out_names = [e[0] for e in libE_fields]
    if EnsembleSpecs.H0 is not None and EnsembleSpecs.H0.dtype.names is not None:
        out_names += list(EnsembleSpecs.H0.dtype.names)
    out_names += [e[0] for e in EnsembleSpecs.sim_specs.outputs]
    if EnsembleSpecs.gen_specs:
        out_names += [e[0] for e in EnsembleSpecs.gen_specs.outputs]
    if EnsembleSpecs.alloc_specs:
        out_names += [e[0] for e in EnsembleSpecs.alloc_specs.outputs]

    for name in EnsembleSpecs.sim_specs.inputs:
        assert name in out_names, (
            name + " in sim_specs['in'] is not in sim_specs['out'], "
            "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
        )

    if EnsembleSpecs.gen_specs:
        for name in EnsembleSpecs.gen_specs.inputs:
            assert name in out_names, (
                name + " in gen_specs['in'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )
    return EnsembleSpecs


def _check_H0(EnsembleSpecs):
    if EnsembleSpecs.H0.size > 0:
        H0 = EnsembleSpecs.H0
        specs = [EnsembleSpecs.sim_specs, EnsembleSpecs.gen_specs]
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
    return EnsembleSpecs


def _check_any_workers_and_disable_rm_if_tcp(LibeSpecs):
    comms_type = LibeSpecs.comms
    if comms_type in ["local", "tcp"]:
        if LibeSpecs.nworkers:
            assert LibeSpecs.nworkers >= 1, "Must specify at least one worker"
        else:
            if comms_type == "tcp":
                assert LibeSpecs.workers, "Without nworkers, must specify worker hosts on TCP"
    if comms_type == "tcp":
        LibeSpecs.__dict__["disable_resource_manager"] = True  # Resource management not supported with TCP
    return LibeSpecs
