import os
import numpy as np
import pytest
import mock

import libensemble.tests.unit_tests.setup as setup
from libensemble.tools.fields_keys import libE_fields
from libensemble.resources.resources import Resources
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.utils.runners import Runners


def get_ufunc_args():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

    L = exit_criteria["sim_max"]
    H = np.zeros(L, dtype=list(set(libE_fields + sim_specs["out"] + gen_specs["out"])))

    H["sim_id"][-L:] = -1
    H["sim_started_time"][-L:] = np.inf

    sim_ids = np.zeros(1, dtype=int)
    Work = {"tag": EVAL_SIM_TAG, "persis_info": {}, "libE_info": {"H_rows": sim_ids}, "H_fields": sim_specs["in"]}
    calc_in = H[Work["H_fields"]][Work["libE_info"]["H_rows"]]
    return calc_in, sim_specs, gen_specs


def test_normal_runners():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    runners = Runners(sim_specs, gen_specs)
    assert not runners.has_funcx_sim and not runners.has_funcx_gen, \
        "funcX use should not be detected without setting endpoint fields"

    ro = runners.make_runners()
    assert all([i in ro for i in [EVAL_SIM_TAG, EVAL_GEN_TAG]]), \
        "Both user function tags should be included in runners dictionary"


def test_normal_no_gen():
    calc_in, sim_specs, gen_specs = get_ufunc_args()

    runners = Runners(sim_specs, {})
    ro = runners.make_runners()

    assert not ro[2], \
        "generator function shouldn't be provided if not using gen_specs"


def test_manager_exception():
    """Checking dump of history and pickle file on abort"""
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    remove_file_if_exists(hfile_abort)
    remove_file_if_exists(pfile_abort)

    with mock.patch("libensemble.manager.manager_main") as managerMock:
        managerMock.side_effect = Exception
        # Collision between libE.py and libE() (after mods to __init__.py) means
        #   libensemble.libE.comms_abort tries to refer to the function, not file
        with mock.patch("libensemble.comms_abort") as abortMock:
            abortMock.side_effect = Exception
            # Need fake MPI to get past the Manager only check and dump history
            with pytest.raises(Exception):
                libE_specs = {"mpi_comm": fake_mpi, "disable_resource_manager": True}
                libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)
                pytest.fail("Expected exception")
            assert os.path.isfile(hfile_abort), "History file not dumped"
            assert os.path.isfile(pfile_abort), "Pickle file not dumped"
            os.remove(hfile_abort)
            os.remove(pfile_abort)

            # Test that History and Pickle files NOT created when disabled
            with pytest.raises(Exception):
                libE_specs = {"mpi_comm": fake_mpi, "save_H_and_persis_on_abort": False}
                libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)
                pytest.fail("Expected exception")
            assert not os.path.isfile(hfile_abort), "History file dumped"
            assert not os.path.isfile(pfile_abort), "Pickle file dumped"




if __name__ == "__main__":
    test_normal_runners()
    test_no_gen()
