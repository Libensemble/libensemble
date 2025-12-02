import numpy as np
from pydantic import ValidationError

import libensemble.tests.unit_tests.setup as setup
from gest_api.vocs import VOCS
from libensemble.gen_funcs.sampling import latin_hypercube_sample
from libensemble.sim_funcs.simple_sim import norm_eval
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


def test_vocs_to_sim_specs():
    """Test that SimSpecs correctly derives inputs and outputs from VOCS"""

    vocs = VOCS(
        variables={"x1": [0, 1], "x2": [0, 10]},
        constants={"c1": 1.0},
        objectives={"y1": "MINIMIZE"},
        observables={"o1": float, "o2": int, "o3": (float, (3,))},
        constraints={"con1": ["GREATER_THAN", 0]},
    )

    ss = SimSpecs(sim_f=norm_eval, vocs=vocs)

    assert ss.inputs == ["x1", "x2", "c1"]
    assert len(ss.outputs) == 5
    output_dict = {}
    for item in ss.outputs:
        if len(item) == 2:
            name, dtype = item
            output_dict[name] = dtype
        else:
            name, dtype, shape = item
            output_dict[name] = (dtype, shape)
    assert output_dict["o1"] == float and output_dict["o2"] == int and output_dict["o3"] == (float, (3,))

    # Explicit values take precedence
    ss2 = SimSpecs(sim_f=norm_eval, vocs=vocs, inputs=["custom"], outputs=[("custom_out", int)])

    assert ss2.inputs == ["custom"] and ss2.outputs == [("custom_out", int)]


def test_vocs_to_gen_specs():
    """Test that GenSpecs correctly derives persis_in and outputs from VOCS"""

    vocs = VOCS(
        variables={"x1": [0, 1], "x2": [0, 10]},
        constants={"c1": 1.0},
        objectives={"y1": "MINIMIZE"},
        observables=["obs1"],
        constraints={"con1": ["GREATER_THAN", 0]},
    )

    gs = GenSpecs(gen_f=latin_hypercube_sample, vocs=vocs)

    assert gs.persis_in == ["x1", "x2", "c1", "y1", "obs1", "con1"]
    assert len(gs.outputs) == 3
    # All default to float if dtype not specified
    for name, dtype in gs.outputs:
        assert dtype == float

    # Explicit values take precedence
    gs2 = GenSpecs(gen_f=latin_hypercube_sample, vocs=vocs, persis_in=["custom"], out=[("custom_out", int)])
    assert gs2.persis_in == ["custom"] and gs2.outputs == [("custom_out", int)]


if __name__ == "__main__":
    test_sim_gen_alloc_exit_specs()
    test_sim_gen_alloc_exit_specs_invalid()
    test_libe_specs()
    test_libe_specs_invalid()
    test_ensemble_specs()
    test_vocs_to_sim_specs()
    test_vocs_to_gen_specs()
