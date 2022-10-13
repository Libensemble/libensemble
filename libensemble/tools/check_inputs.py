import os
import numpy as np
import logging
from libensemble.tools.fields_keys import (
    libE_fields,
    allowed_gen_spec_keys,
    allowed_sim_spec_keys,
    allowed_alloc_spec_keys,
    allowed_libE_spec_keys,
)

logger = logging.getLogger(__name__)


def _check_consistent_field(name, field0, field1):
    """Checks that new field (field1) is compatible with an old field (field0)."""
    assert field0.ndim == field1.ndim, f"H0 and H have different ndim for field {name}"
    assert np.all(
        np.array(field1.shape) >= np.array(field0.shape)
    ), f"H too small to receive all components of H0 in field {name}"


def check_libE_specs(libE_specs, serial_check=False):
    assert isinstance(libE_specs, dict), "libE_specs must be a dictionary"

    comms_type = libE_specs.get("comms", "mpi")
    if comms_type in ["mpi"]:
        if not serial_check:
            assert libE_specs["mpi_comm"].Get_size() > 1, "Manager only - must be at least one worker (2 MPI tasks)"
    elif comms_type in ["local"]:
        assert libE_specs["nworkers"] >= 1, "Must specify at least one worker"
    elif comms_type in ["tcp"]:
        # TODO, differentiate and test SSH/Client
        assert libE_specs["nworkers"] >= 1, "Must specify at least one worker"

    for k in libE_specs.keys():
        assert k in allowed_libE_spec_keys, "Key %s is not allowed in libE_specs. Supported keys are: %s " % (
            k,
            allowed_libE_spec_keys,
        )

        if k in ["ensemble_copy_back", "use_worker_dirs", "sim_dirs_make", "gen_dirs_make"]:
            assert isinstance(libE_specs[k], bool), f"Value for libE_specs['{k}'] must be boolean"

        if k in ["sim_input_dir", "gen_input_dir"]:
            assert isinstance(libE_specs[k], str), f"Value for libE_specs['{k}'] must be a single path-like string"
            assert os.path.exists(libE_specs[k]), f"libE_specs['{k}'] does not refer to an existing path."

        if k == "ensemble_dir_path":
            assert isinstance(libE_specs[k], str), f"Value for libE_specs['{k}'] must be a single path-like string"

        if k in ["sim_dir_copy_files", "sim_dir_symlink_files", "gen_dir_copy_files", "gen_dir_symlink_files"]:
            assert isinstance(libE_specs[k], list), f"Value for libE_specs['{k}'] must be a list of path-like strings"
            for j in libE_specs[k]:
                assert os.path.exists(j), f"'{j}' in libE_specs['{k}'] does not refer to an existing path."


def check_alloc_specs(alloc_specs):
    assert isinstance(alloc_specs, dict), "alloc_specs must be a dictionary"

    assert alloc_specs["alloc_f"], "Allocation function must be specified"

    for k in alloc_specs.keys():
        assert k in allowed_alloc_spec_keys, "Key %s is not allowed in alloc_specs. Supported keys are: %s " % (
            k,
            allowed_alloc_spec_keys,
        )


def check_sim_specs(sim_specs):
    assert isinstance(sim_specs, dict), "sim_specs must be a dictionary"

    assert any(
        [term_field in sim_specs for term_field in ["sim_f", "in", "persis_in", "out"]]
    ), "sim_specs must contain 'sim_f', 'in', 'out'"

    assert all(
        isinstance(i, str) for i in sim_specs["in"]
    ), "Entries in sim_specs['in'] must be strings. Also can't be lists or tuples of strings."

    assert len(sim_specs["out"]), "sim_specs must have 'out' entries"

    assert isinstance(sim_specs["in"], list), "'in' field must exist and be a list of field names"

    for k in sim_specs.keys():
        assert k in allowed_sim_spec_keys, "Key %s is not allowed in sim_specs. Supported keys are: %s " % (
            k,
            allowed_sim_spec_keys,
        )


def check_gen_specs(gen_specs):
    assert isinstance(gen_specs, dict), "gen_specs must be a dictionary"

    assert not bool(gen_specs) or len(gen_specs["out"]), "gen_specs must have 'out' entries"

    if "in" in gen_specs:
        assert all(
            isinstance(i, str) for i in gen_specs["in"]
        ), "Entries in gen_specs['in'] must be strings. Also can't be lists or tuples of strings."

    for k in gen_specs.keys():
        assert k in allowed_gen_spec_keys, "Key %s is not allowed in gen_specs. Supported keys are: %s " % (
            k,
            allowed_gen_spec_keys,
        )


def check_exit_criteria(exit_criteria, sim_specs, gen_specs):
    assert isinstance(exit_criteria, dict), "exit_criteria must be a dictionary"

    assert len(exit_criteria) > 0, "Must have some exit criterion"

    if "elapsed_wallclock_time" in exit_criteria:
        logger.warning(
            "exit_criteria['elapsed_wallclock_time'] is deprecated.'\n"
            + "This will break in the future. Use exit_criteria['wallclock_max']"
        )

        exit_criteria["wallclock_max"] = exit_criteria.pop("elapsed_wallclock_time")

    # Ensure termination criteria are valid
    valid_term_fields = ["sim_max", "gen_max", "wallclock_max", "stop_val"]
    assert all([term_field in valid_term_fields for term_field in exit_criteria]), "Valid termination options: " + str(
        valid_term_fields
    )

    # Make sure stop-values match parameters in gen_specs or sim_specs
    if "stop_val" in exit_criteria:
        stop_name = exit_criteria["stop_val"][0]
        sim_out_names = [e[0] for e in sim_specs["out"]]
        gen_out_names = [e[0] for e in gen_specs["out"]]
        assert stop_name in sim_out_names + gen_out_names, "Can't stop on {} if it's not in a sim/gen output".format(
            stop_name
        )


def check_H(H0, sim_specs, alloc_specs, gen_specs):
    if len(H0):
        # Set up dummy history to see if it agrees with H0

        # Combines all 'out' fields (if they exist) in sim_specs, gen_specs, or alloc_specs
        specs = [sim_specs, alloc_specs, gen_specs]
        dtype_list = list(set(libE_fields + sum([k.get("out", []) for k in specs if k], [])))
        Dummy_H = np.zeros(1 + len(H0), dtype=dtype_list)

        fields = H0.dtype.names

        # Prior history must contain the fields in new history
        assert set(fields).issubset(set(Dummy_H.dtype.names)), "H0 contains fields {} not in the History.".format(
            set(fields).difference(set(Dummy_H.dtype.names))
        )

        # Prior history cannot contain unreturned points
        # assert 'sim_ended' not in fields or np.all(H0['sim_ended']), \
        #     "H0 contains unreturned points."

        # Fail if prior history contains unreturned points (or returned but not given).
        assert "sim_ended" not in fields or np.all(
            H0["sim_started"] == H0["sim_ended"]
        ), "H0 contains unreturned or invalid points"

        # # Fail if points in prior history don't have a sim_id.
        # assert('sim_id' in fields), 'Points in H0 must have sim_ids'

        # Check dimensional compatibility of fields
        for field in fields:
            _check_consistent_field(field, H0[field], Dummy_H[field])


def check_inputs(
    libE_specs=None, alloc_specs=None, sim_specs=None, gen_specs=None, exit_criteria=None, H0=None, serial_check=False
):
    """
    Checks whether the libEnsemble arguments are of the correct data type and
    contain sufficient information to perform a run. There is no return value.
    An exception is raised if any of the checks fail.

    .. code-block:: python

        from libensemble.tools import check_inputs
        check_inputs(sim_specs=my_sim_specs, gen_specs=my_gen_specs, exit_criteria=ec)

    Parameters
    ----------

    libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria: :obj:`dict`, optional

        libEnsemble data structures

    H0: :obj:`numpy structured array`, optional

        A previous libEnsemble history to be prepended to the history in the
        current libEnsemble run
        :doc:`(example)<data_structures/history_array>`

    serial_check : :obj:`boolean`

        If true, assumes running a serial check. This means, for example,
        the details of current MPI communicator are not checked (can be
        run with libE_specs{'mpi_comm': 'mpi'} without running through mpiexec.

    """
    out_names = [e[0] for e in libE_fields]
    if H0 is not None and H0.dtype.names is not None:
        out_names += list(H0.dtype.names)
    if sim_specs is not None:
        out_names += [e[0] for e in sim_specs.get("out", [])]
    if gen_specs is not None:
        out_names += [e[0] for e in gen_specs.get("out", [])]
    if alloc_specs is not None:
        out_names += [e[0] for e in alloc_specs.get("out", [])]

    # Detailed checking based on Required Keys in docs for each specs
    if libE_specs is not None:
        for name in libE_specs.get("final_fields", []):
            assert name in out_names, (
                name + " in libE_specs['fields_keys'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )
        check_libE_specs(libE_specs, serial_check)

    if alloc_specs is not None:
        assert "in" not in alloc_specs, "alloc_specs['in'] is not needed as all of the history is available alloc_f."

        check_alloc_specs(alloc_specs)

    if sim_specs is not None:
        if "in" in sim_specs:
            assert isinstance(sim_specs["in"], list), "sim_specs['in'] must be a list"

        for name in sim_specs.get("in", []):
            assert name in out_names, (
                name + " in sim_specs['in'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )

        check_sim_specs(sim_specs)

    if gen_specs is not None:
        if "in" in gen_specs:
            assert isinstance(gen_specs["in"], list), "gen_specs['in'] must be a list"

        for name in gen_specs.get("in", []):
            assert name in out_names, (
                name + " in gen_specs['in'] is not in sim_specs['out'], "
                "gen_specs['out'], alloc_specs['out'], H0, or libE_fields."
            )

        check_gen_specs(gen_specs)

    if exit_criteria is not None:
        assert (
            sim_specs is not None and gen_specs is not None
        ), "Can't check exit_criteria without sim_specs and gen_specs"
        check_exit_criteria(exit_criteria, sim_specs, gen_specs)

    if H0 is not None:
        assert (
            sim_specs is not None and alloc_specs is not None and gen_specs is not None
        ), "Can't check H0 without sim_specs, alloc_specs, gen_specs"
        check_H(H0, sim_specs, alloc_specs, gen_specs)
