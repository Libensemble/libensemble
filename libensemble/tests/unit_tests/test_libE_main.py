import os
import numpy as np
import pytest
import mock

from libensemble.libE import check_inputs, libE
import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from mpi4py import MPI
from libensemble.tests.regression_tests.common import mpi_comm_excl
from libensemble.comms.logs import LogConfig
from numpy import inf


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
        assert flag == 1, 'Aborting without exit code of 1'
        raise MPIAbortException()


fake_mpi = Fake_MPI()

libE_specs = {'comm': MPI.COMM_WORLD}
alloc_specs = {'alloc_f': give_sim_work_first, 'out': [('allocated', bool)]}
hfile_abort = 'libE_history_at_abort_0.npy'
pfile_abort = 'libE_persis_info_at_abort_0.pickle'


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

    with mock.patch('libensemble.libE_manager.manager_main') as managerMock:
        managerMock.side_effect = Exception
        # Collision between libE.py and libE() (after mods to __init__.py) means
        #   libensemble.libE.comms_abort tries to refer to the function, not file
        with mock.patch('libensemble.comms_abort') as abortMock:
            abortMock.side_effect = Exception
            # Need fake MPI to get past the Manager only check and dump history
            with pytest.raises(Exception):
                libE(sim_specs, gen_specs, exit_criteria, libE_specs={'comm': fake_mpi})
                pytest.fail('Expected exception')
            assert os.path.isfile(hfile_abort), "History file not dumped"
            assert os.path.isfile(pfile_abort), "Pickle file not dumped"
            os.remove(hfile_abort)
            os.remove(pfile_abort)

            # Test that History and Pickle files NOT created when disabled
            with pytest.raises(Exception):
                libE(sim_specs, gen_specs, exit_criteria,
                     libE_specs={'comm': fake_mpi, 'save_H_and_persis_on_abort': False})
                pytest.fail('Expected exception')
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
        libE(sim_specs, gen_specs, exit_criteria, libE_specs={'comm': fake_mpi})
        pytest.fail('Expected MPIAbortException exception')


def test_exception_raising_manager_no_abort():
    """Running until fake_MPI tries to send msg to test (mocked) comm.Abort is called

    Manager should raise MPISendException when fakeMPI tries to send message, which
    will be caught by libE and raise MPIAbortException from fakeMPI.Abort"""
    libE_specs['abort_on_exception'] = False
    libE_specs['comm'] = fake_mpi
    with pytest.raises(MPISendException):
        sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
        libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)
        pytest.fail('Expected MPISendException exception')


# So it's a key error rather than assertion error as does not test if 'in' is
# missing, only that its a list - needs updating in future.
def test_exception_raising_check_inputs():
    """Intentionally running without sim_specs['in'] to test exception raising (Fails)"""
    with pytest.raises(KeyError):
        H, _, _ = libE({'out': [('f', float)]}, {'out': [('x', float)]}, {'sim_max': 1}, libE_specs={'comm': fake_mpi})
        pytest.fail('Expected KeyError exception')


def test_proc_not_in_communicator():
    """Checking proc not in communicator returns exit status of 3"""
    libE_specs['comm'], mpi_comm_null = mpi_comm_excl()
    H, _, flag = libE({'in': ['x'], 'out': [('f', float)]}, {'out': [('x', float)]},
                      {'sim_max': 1}, libE_specs=libE_specs)
    assert flag == 3, "libE return flag should be 3. Returned: " + str(flag)


# def test_exception_raising_worker():
#     # Intentionally running without sim_specs['in'] to test exception raising (Fails)
#     H, _, _ = libE({'out': [('f', float)]}, {'out': [('x', float)]},
#                    {'sim_max': 1}, libE_specs={'comm': MPI.COMM_WORLD})
#     assert H==[]


def rmfield(a, *fieldnames_to_remove):
    return a[[name for name in a.dtype.names if name not in fieldnames_to_remove]]


def check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0):
    with pytest.raises(AssertionError) as excinfo:
        check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
        pytest.fail('Expected AssertionError exception')
    return str(excinfo.value)


def test_checking_inputs_noworkers():
    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    H0 = np.empty(0)

    # Should fail because only got a manager
    libE_specs = {'comm': MPI.COMM_WORLD, 'comms': 'mpi'}
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert 'must be at least one worker' in errstr, 'Incorrect assertion error: ' + errstr


def test_checking_inputs_H0():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {'comm': fake_mpi, 'comms': 'mpi'}

    # Should fail because H0 has points with 'return'==False
    H0 = np.array([(False, 0., 0, 0., 1, True, 1, True, [0., 0., 0.], True, 0.1, 1.1),
                   (False, 0., 0, 0., 1, True, 2, True, [0., 0., 0.], True, 0.2, 1.2),
                   (False, 0., 0, 0., 1, True, 3, True, [0., 0., 0.], True, 0.3, 1.3),
                   (False, 0., 0, 0., -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                   (False, 0., 0, 0., -1, False, 0, False, [0., 0., 0.], False, 0., inf)],
                  dtype=[('local_pt', '?'), ('priority', '<f8'), ('gen_worker', '<i8'),
                         ('x_on_cube', '<f8'), ('sim_id', '<i8'), ('given', '?'),
                         ('sim_worker', '<i8'), ('returned', '?'), ('fvec', '<f8', (3,)),
                         ('allocated', '?'), ('f', '<f8'), ('given_time', '<f8')])

    # This should work
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # A value has not been returned
    H0['returned'][2] = False
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert 'H0 contains unreturned or invalid points' in errstr, 'Incorrect assertion error: ' + errstr

    # Ungiven points shown as returned
    H0['returned'] = True
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert 'H0 contains unreturned or invalid points' in errstr, 'Incorrect assertion error: ' + errstr

    # Return to correct state
    H0['returned'][3:len(H0)] = False
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Removing 'returned' and then testing again. Should be successful as 'returned' does not exist
    H0 = rmfield(H0, 'returned')
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Should fail because H0 has fields not in H
    H0 = np.zeros(3, dtype=sim_specs['out'] + gen_specs['out'] + alloc_specs['out'] + [('bad_name2', bool)])
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert 'not in the History' in errstr, 'Incorrect assertion error: ' + errstr


def test_checking_inputs_exit_crit():
    sim_specs, gen_specs, _ = setup.make_criteria_and_specs_0()
    libE_specs = {'comm': fake_mpi, 'comms': 'mpi'}
    H0 = np.empty(0)

    exit_criteria = {}
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert 'Must have some exit criterion' in errstr, 'Incorrect assertion error: ' + errstr

    exit_criteria = {'swim_max': 10}
    errstr = check_assertion(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    assert 'Valid termination options' in errstr, 'Incorrect assertion error: ' + errstr


def test_checking_inputs_single():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {'comm': fake_mpi, 'comms': 'mpi'}

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


def test_logging_disabling():
    remove_file_if_exists('ensemble.log')
    remove_file_if_exists('libE_stats.txt')
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()
    libE_specs = {'comm': fake_mpi, 'comms': 'mpi', 'disable_log_files': True}
    logconfig = LogConfig.config
    logconfig.logger_set = False

    with mock.patch('libensemble.libE_manager.manager_main') as managerMock:
        managerMock.side_effect = Exception
        with mock.patch('libensemble.comms_abort') as abortMock:
            abortMock.side_effect = Exception
            with pytest.raises(Exception):
                libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)
                pytest.fail('Expected exception')
            assert not os.path.isfile('ensemble.log'), "ensemble.log file dumped"
            assert not os.path.isfile('libE_stats.txt'), "libE_stats.txt file dumped"


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
