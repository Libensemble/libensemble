import os
import numpy as np
import pytest
import mock

from libensemble.libE import check_inputs, libE
from libensemble.manager import LoggedException
import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.tools.fields_keys import libE_fields
from libensemble.resources.resources import Resources
from libensemble.tests.regression_tests.common import mpi_comm_excl
from libensemble.comms.logs import LogConfig


class MPIAbortException(Exception):
    "Raised when mock mpi abort is called"


class MPISendException(Exception):
    "Raised when mock mpi abort is called"


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

    def isend(self, msg, dest, tag):
        raise MPISendException()

    def Abort(self, flag):
        assert flag == 1, "Aborting without exit code of 1"
        raise MPIAbortException()


class Fake_MPI_1P(Fake_MPI):
    def Get_size(self):
        return 1


fake_mpi = Fake_MPI()
fake_mpi_1p = Fake_MPI_1P()

alloc_specs = {"alloc_f": give_sim_work_first, "out": []}
hfile_abort = "libE_history_at_abort_0.npy"
pfile_abort = "libE_persis_info_at_abort_0.pickle"


# Run by pytest before each function
def setup_function(function):
    print(f"setup_function function:{function.__name__}")
    if Resources.resources is not None:
        del Resources.resources
        Resources.resources = None


def remove_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


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


# So it's a key error rather than assertion error as does not test if 'in' is
# missing, only that it's a list - needs updating in future.
def test_exception_raising_check_inputs():
    """Intentionally running without sim_specs['in'] to test exception raising (Fails)"""
    libE_specs = {"mpi_comm": fake_mpi, "disable_resource_manager": True}
    with pytest.raises(KeyError):
        H, _, _ = libE({"out": [("f", float)]}, {"out": [("x", float)]}, {"sim_max": 1}, libE_specs=libE_specs)
        pytest.fail("Expected KeyError exception")


def test_proc_not_in_communicator():
    """Checking proc not in communicator returns exit status of 3"""
    libE_specs = {}
    libE_specs["mpi_comm"], mpi_comm_null = mpi_comm_excl()
    H, _, flag = libE(
        {"in": ["x"], "out": [("f", float)]}, {"out": [("x", float)]}, {"sim_max": 1}, libE_specs=libE_specs
    )
    assert flag == 3, "libE return flag should be 3. Returned: " + str(flag)


# def test_exception_raising_worker():
#     # Intentionally running without sim_specs['in'] to test exception raising (Fails)
#     H, _, _ = libE({'out': [('f', float)]}, {'out': [('x', float)]},
#                    {'sim_max': 1}, libE_specs={'mpi_comm': MPI.COMM_WORLD})
#     assert H==[]


def rmfield(a, *fieldnames_to_remove):
    return a[[name for name in a.dtype.names if name not in fieldnames_to_remove]]


def check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0):
    with pytest.raises(AssertionError) as excinfo:
        check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
        pytest.fail("Expected AssertionError exception")
    return str(excinfo.value)


def test_checking_inputs_noworkers():
    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    H0 = np.empty(0)
    # Should fail because only got a manager
    libE_specs = {"mpi_comm": fake_mpi_1p, "comms": "mpi"}
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert "must be at least one worker" in errstr, "Incorrect assertion error: " + errstr


def test_checking_inputs_H0():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {"mpi_comm": fake_mpi, "comms": "mpi"}

    # Should fail because H0 has points with 'sim_ended'==False
    H0 = np.zeros(5, dtype=libE_fields)
    H0["sim_id"] = [0, 1, 2, -1, -1]
    H0["sim_worker"][0:3] = range(1, 4)
    H0[["sim_started", "sim_ended"]][0:3] = True

    # This should work
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # A value has not been marked as sim_ended
    H0["sim_ended"][2] = False
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert "H0 contains unreturned or invalid points" in errstr, "Incorrect assertion error: " + errstr

    # Points that have not been marked as 'sim_started' have been marked 'sim_ended'
    H0["sim_ended"] = True
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert "H0 contains unreturned or invalid points" in errstr, "Incorrect assertion error: " + errstr

    # Return to correct state
    H0["sim_ended"][3 : len(H0)] = False
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Removing 'sim_ended' and then testing again. Should be successful as 'sim_ended' does not exist
    H0 = rmfield(H0, "sim_ended")
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Should fail because H0 has fields not in H
    H0 = np.zeros(3, dtype=sim_specs["out"] + gen_specs["out"] + alloc_specs["out"] + [("bad_name2", bool)])
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert "not in the History" in errstr, "Incorrect assertion error: " + errstr


def test_checking_inputs_exit_crit():
    sim_specs, gen_specs, _ = setup.make_criteria_and_specs_0()
    libE_specs = {"mpi_comm": fake_mpi, "comms": "mpi"}
    H0 = np.empty(0)

    exit_criteria = {}
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert "Must have some exit criterion" in errstr, "Incorrect assertion error: " + errstr

    exit_criteria = {"swim_max": 10}
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert "Valid termination options" in errstr, "Incorrect assertion error: " + errstr


def test_checking_inputs_single():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {"mpi_comm": fake_mpi, "comms": "mpi"}

    check_inputs(libE_specs=libE_specs)
    check_inputs(alloc_specs=alloc_specs)
    try:
        check_inputs(sim_specs=sim_specs)
    except AssertionError:
        assert 1, "Fails because sim_specs['in']=['x_on_cube'] and that's not an 'out' of anything"
    else:
        assert 0, "Should have failed"
    check_inputs(gen_specs=gen_specs)
    check_inputs(exit_criteria=exit_criteria, sim_specs=sim_specs, gen_specs=gen_specs)

    libE_specs["use_worker_dirs"] = True
    libE_specs["sim_input_dir"] = "./__init__.py"
    libE_specs["sim_dir_copy_files"] = ["./__init__.py"]
    check_inputs(libE_specs=libE_specs)


@pytest.mark.extra
def test_logging_disabling():
    remove_file_if_exists("ensemble.log")
    remove_file_if_exists("libE_stats.txt")
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {"mpi_comm": fake_mpi, "comms": "mpi", "disable_log_files": True}
    logconfig = LogConfig.config
    logconfig.logger_set = False

    with mock.patch("libensemble.manager.manager_main") as managerMock:
        managerMock.side_effect = Exception
        with mock.patch("libensemble.comms_abort") as abortMock:
            abortMock.side_effect = Exception
            with pytest.raises(Exception):
                libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)
                pytest.fail("Expected exception")
            assert not os.path.isfile("ensemble.log"), "ensemble.log file dumped"
            assert not os.path.isfile("libE_stats.txt"), "libE_stats.txt file dumped"


if __name__ == "__main__":
    test_manager_exception()
    test_exception_raising_manager_with_abort()
    test_exception_raising_manager_no_abort()
    test_exception_raising_check_inputs()
    test_proc_not_in_communicator()
    test_checking_inputs_noworkers()
    test_checking_inputs_H0()
    test_checking_inputs_exit_crit()
    test_checking_inputs_single()
    test_logging_disabling()
