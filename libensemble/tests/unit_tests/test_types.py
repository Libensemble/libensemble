import numpy as np

import libensemble.tests.unit_tests.setup as setup
from libensemble.tools.fields_keys import libE_fields
from libensemble.types import SimSpecs, GenSpecs, ExitCriteria, LibeSpecs, EnsembleSpecs


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
    ss = SimSpecs.parse_obj(sim_specs)
    gs = GenSpecs.parse_obj(gen_specs)
    ec = ExitCriteria.parse_obj(exit_criteria)

    # maybe I don't need to test this - this should be taken for granted as part of pydantic?
    assert ss.dict(by_alias=True, exclude_unset=True) == sim_specs, "sim_specs model conversion failed"
    assert gs.dict(by_alias=True, exclude_unset=True) == gen_specs, "gen_specs model conversion failed"
    assert ec.dict(by_alias=True, exclude_unset=True) == exit_criteria, "exit_criteria model conversion failed"


def test_libe_specs():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {"mpi_comm": Fake_MPI(), "comms": "mpi"}
    ls = LibeSpecs.parse_obj(libE_specs)

    libE_specs["sim_input_dir"] = "./simdir"
    libE_specs["sim_dir_copy_files"] = ["./simdir"]
    ls = LibeSpecs.parse_obj(libE_specs)

    ls = LibeSpecs.parse_obj({"comms": "tcp", "nworkers": 4})
    assert ls.disable_resource_manager, "resource manager should be disabled when using tcp comms"
    ls = LibeSpecs.parse_obj({"comms": "tcp", "workers": ["hello.host"]})


def test_ensemble_specs():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {"comms": "local", "nworkers": 4}
    ss = SimSpecs.parse_obj(sim_specs)
    gs = GenSpecs.parse_obj(gen_specs)
    ec = ExitCriteria.parse_obj(exit_criteria)
    ls = LibeSpecs.parse_obj(libE_specs)

    # Should fail because H0 has points with 'sim_ended'==False
    H0 = np.zeros(5, dtype=libE_fields)
    H0["sim_id"] = [0, 1, 2, -1, -1]
    H0["sim_worker"][0:3] = range(1, 4)
    H0[["sim_started", "sim_ended"]][0:3] = True

    es = EnsembleSpecs(H0=H0, libE_specs=ls, sim_specs=ss, gen_specs=gs, exit_criteria=ec)
    assert es.nworkers == libE_specs["nworkers"], "nworkers not passed through to ensemble specs"


if __name__ == "__main__":
    test_sim_gen_alloc_exit_specs()
    test_libe_specs()
    test_ensemble_specs()
