import os

import mock
import pytest

import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.comms.logs import LogConfig
from libensemble.libE import libE
from libensemble.manager import LoggedException
from libensemble.resources.resources import Resources
from libensemble.tests.regression_tests.common import mpi_comm_excl


class MPIAbortException(Exception):
    """Raised when mock mpi abort is called"""


class MPISendException(Exception):
    """Raised when mock mpi abort is called"""


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

alloc_specs = {"alloc_f": give_sim_work_first}
hfile_abort = "libE_history_at_abort_0.npy"
pfile_abort = "libE_persis_info_at_abort_0.pickle"


# Run by pytest at end of module
def teardown_module(module):
    try:
        print(f"teardown_module module:{module.__name__}")
    except AttributeError:
        print(f"teardown_module (direct run) module:{module}")
    if Resources.resources is not None:
        del Resources.resources
        Resources.resources = None


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
        with mock.patch("libensemble.libE.comms_abort") as abortMock:
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


# So it's a key error rather than assertion error as does not test if "in" is
# missing, only that it's a list - needs updating in future.
def test_exception_raising_check_inputs():
    """Intentionally running without sim_specs["in"] to test exception raising (Fails)"""
    libE_specs = {"mpi_comm": fake_mpi, "disable_resource_manager": True}
    with pytest.raises(Exception):
        H, _, _ = libE({"out": [("f", float)]}, {"out": [("x", float)]}, {"sim_max": 1}, libE_specs=libE_specs)
        pytest.fail("Expected ValidationError exception")


def test_proc_not_in_communicator():
    """Checking proc not in communicator returns exit status of 3"""
    libE_specs = {}
    libE_specs["mpi_comm"], mpi_comm_null = mpi_comm_excl()
    H, _, flag = libE(
        {"in": ["x"], "out": [("f", float)]}, {"out": [("x", float)]}, {"sim_max": 1}, libE_specs=libE_specs
    )
    assert flag == 3, "libE return flag should be 3. Returned: " + str(flag)


# def test_exception_raising_worker():
#     # Intentionally running without sim_specs["in"] to test exception raising (Fails)
#     H, _, _ = libE({"out": [("f", float)]}, {"out": [("x", float)]},
#                    {"sim_max": 1}, libE_specs={"mpi_comm": MPI.COMM_WORLD})
#     assert H==[]


def rmfield(a, *fieldnames_to_remove):
    return a[[name for name in a.dtype.names if name not in fieldnames_to_remove]]


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
        with mock.patch("libensemble.libE.comms_abort") as abortMock:
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
    test_logging_disabling()
