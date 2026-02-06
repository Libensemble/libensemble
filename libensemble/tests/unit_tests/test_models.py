import numpy as np
from pydantic import ValidationError

import libensemble.tests.unit_tests.setup as setup
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs, _EnsembleSpecs
from libensemble.utils.misc import specs_dump


class Fake_MPI:
    """Explicit mocking of MPI communicator"""

    def Get_size(self):
        return 2

    def Get_rank(self):
        return 0

    def Barrier(self):
        return 0

    def Dup(self):
        return self

    def Free(self):
        return


def test_sim_gen_alloc_exit_specs():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    ss = SimSpecs(**sim_specs)
    gs = GenSpecs(**gen_specs)
    ec = ExitCriteria(**exit_criteria)

    # maybe I don't need to test this - this should be taken for granted as part of pydantic?
    assert specs_dump(ss, by_alias=True, exclude_unset=True) == sim_specs, "sim_specs model conversion failed"
    assert specs_dump(gs, by_alias=True, exclude_unset=True) == gen_specs, "gen_specs model conversion failed"
    assert specs_dump(ec, by_alias=True, exclude_unset=True) == exit_criteria, "exit_criteria model conversion failed"


def test_sim_gen_alloc_exit_specs_invalid():
    bad_specs = {
        "sim_f": "path.to.module",  # sim: _UFUNC_INVALID_ERR, gen: _UNRECOGNIZED_ERR
        "in": [("f", float), ("fvec", float, 3)],  # _IN_INVALID_ERR
        "out": ["x_on_cube"],  # _OUT_DTYPE_ERR
        "globus_compute_endpoint": 1234,  # invalid endpoint, should be str
        "user": np.zeros(10),  # 'value is not a valid dict'
    }

    try:
        SimSpecs.model_validate(bad_specs)
        flag = 0
    except ValidationError as e:
        assert len(e.errors()) > 1, "SimSpecs model should have detected multiple errors in specs"
        flag = 1
    assert flag, "SimSpecs didn't raise ValidationError on invalid specs"

    try:
        GenSpecs.model_validate(bad_specs)
        flag = 0
    except ValidationError as e:
        assert len(e.errors()) > 1, "Should've detected multiple errors in specs"
        flag = 1
    assert flag, "GenSpecs didn't raise ValidationError on invalid specs"

    bad_ec = {"stop_vals": 0.5}

    try:
        ExitCriteria.model_validate(bad_ec)
        flag = 0
    except ValidationError:
        flag = 1
    assert flag, "ExitCriteria didn't raise ValidationError on invalid specs"


def test_libe_specs():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {"mpi_comm": Fake_MPI(), "comms": "mpi"}
    ls = LibeSpecs.model_validate(libE_specs)

    libE_specs["sim_input_dir"] = "./simdir"
    libE_specs["sim_dir_copy_files"] = ["./simdir"]
    ls = LibeSpecs.model_validate(libE_specs)

    libE_specs = {"comms": "tcp", "nworkers": 4}

    ls = LibeSpecs.model_validate(libE_specs)
    assert ls.disable_resource_manager, "resource manager should be disabled when using tcp comms"

    libE_specs = {"comms": "tcp", "workers": ["hello.host"]}
    ls = LibeSpecs.model_validate(libE_specs)


def test_libe_specs_invalid():
    bad_specs = {"comms": "local", "zero_resource_workers": 2, "sim_input_dirs": ["obj"]}

    try:
        LibeSpecs.model_validate(bad_specs)
        flag = 0
    except ValidationError:
        flag = 1
    assert flag, "LibeSpecs didn't raise ValidationError on invalid specs"


def test_ensemble_specs():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {"comms": "local", "nworkers": 4}
    ss = SimSpecs(**sim_specs)
    gs = GenSpecs(**gen_specs)
    ec = ExitCriteria(**exit_criteria)
    ls = LibeSpecs(**libE_specs)

    H0 = np.zeros(5, dtype=[("x_on_cube", float, 8), ("sim_id", int), ("sim_started", bool), ("sim_ended", bool)])
    H0["sim_id"] = [0, 1, 2, -1, -1]
    H0[["sim_started", "sim_ended"]][0:3] = True

    _EnsembleSpecs(H0=H0, libE_specs=ls, sim_specs=ss, gen_specs=gs, exit_criteria=ec)


if __name__ == "__main__":
    test_sim_gen_alloc_exit_specs()
    test_sim_gen_alloc_exit_specs_invalid()
    test_libe_specs()
    test_libe_specs_invalid()
    test_ensemble_specs()
