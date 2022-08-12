import os
import numpy as np
import pytest
import mock

import libensemble.tests.unit_tests.setup as setup
from libensemble.tools.fields_keys import libE_fields
from libensemble.resources.resources import Resources
from libensemble.utils import runners


def test_normal_runners():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()




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


# Note - this could be combined now with above tests as fake_MPI prevents need for use of mock module
# Only way that is better is that this will simply hit first code exception - (when fake_MPI tries to isend)
# While first test triggers on call to manager
def test_exception_raising_manager_with_abort():
    """Running until fake_MPI tries to send msg to test (mocked) comm.Abort is called

    Manager should raise MPISendException when fakeMPI tries to send message, which
    will be caught by libE and raise MPIAbortException from fakeMPI.Abort"""
    with pytest.raises(MPIAbortException):
        sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
        libE_specs = {"mpi_comm": fake_mpi, "disable_resource_manager": True}
        libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)
        pytest.fail("Expected MPIAbortException exception")


def test_exception_raising_manager_no_abort():
    """Running until fake_MPI tries to send msg to test (mocked) comm.Abort is called

    Manager should raise MPISendException when fakeMPI tries to send message, which
    will be caught by libE and raise MPIAbortException from fakeMPI.Abort"""
    libE_specs = {"abort_on_exception": False, "mpi_comm": fake_mpi, "disable_resource_manager": True}
    with pytest.raises(LoggedException):
        sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
        libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)
        pytest.fail("Expected MPISendException exception")



if __name__ == "__main__":
    test_manager_exception()
    test_exception_raising_manager_with_abort()
    test_exception_raising_manager_no_abort()