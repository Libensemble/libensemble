import os
import numpy as np
import pytest
import mock

from libensemble.libE import check_inputs, libE
import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from mpi4py import MPI
from libensemble.tests.regression_tests.common import mpi_comm_excl


class Fake_MPI:
    def Get_size(self):
        return 2

    def Get_rank(self):
        return 0

    def Barrier(self):
        return 0


fake_mpi = Fake_MPI()

libE_specs = {'comm': MPI.COMM_WORLD}
alloc_specs = {'alloc_f': give_sim_work_first, 'out': [('allocated', bool)]}
hfile_abort = 'libE_history_at_abort_0.npy'
pfile_abort = 'libE_history_at_abort_0.pickle'


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

    with mock.patch('libensemble.libE.manager_main') as managerMock:
        managerMock.side_effect = Exception
        with mock.patch('libensemble.libE.comms_abort') as abortMock:
            abortMock.side_effect = Exception
            # Need fake MPI to get past the Manager only check and dump history
            with pytest.raises(Exception):
                libE(sim_specs, gen_specs, exit_criteria, libE_specs={'comm': fake_mpi})
                pytest.fail('Expected exception')
            assert os.path.isfile(hfile_abort), "History file not dumped"
            assert os.path.isfile(pfile_abort), "Pickle file not dumped"
            os.remove(hfile_abort)
            os.remove(pfile_abort)


def test_exception_raising_manager():
    """Intentionally running without sim_specs['in'] to test exception raising (Fails)"""
    with mock.patch('libensemble.libE.comms_abort') as abortMock:
        abortMock.side_effect = Exception
        with pytest.raises(KeyError):
            H, _, _ = libE({'out': [('f', float)]}, {'out': [('x', float)]}, {'sim_max': 1}, libE_specs={'comm': fake_mpi})
            pytest.fail('Expected KeyError exception')


def test_proc_not_in_communicator():
    """Checking proc not in communicator returns exit status of 3"""
    libE_specs['comm'], mpi_comm_null = mpi_comm_excl()
    H, _, flag = libE({'in': ['x'], 'out': [('f', float)]}, {'out': [('x', float)]}, {'sim_max': 1}, libE_specs=libE_specs)
    assert flag == 3, "libE return flag should be 3. Returned: " + str(flag)


# def test_exception_raising_worker():
#     # Intentionally running without sim_specs['in'] to test exception raising (Fails)
#     H, _, _ = libE({'out': [('f', float)]}, {'out': [('x', float)]}, {'sim_max': 1}, libE_specs={'comm': MPI.COMM_WORLD})
#     assert H==[]

def test_checking_inputs():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

    # Should fail because only got a manager
    H0 = {}
    libE_specs = {'comm': MPI.COMM_WORLD, 'comms': 'mpi'}
    try:
        check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    except AssertionError:
        assert 1
    else:
        assert 0

    libE_specs['comm'] = fake_mpi

    # Test warning for unreturned points
    H0 = np.zeros(3, dtype=sim_specs['out'] + gen_specs['out'] +
                  alloc_specs['out'] + [('returned', bool)])
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # # Should fail because H0 has points with 'return'==False
    # try:
    #     check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    # except AssertionError:
    #     assert 1
    # else:
    #     assert 0

    # # Should not fail
    # H0['returned'] = True
    # check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    # #
    # # Removing 'returned' and then testing again.
    # H0 = rmfield(H0, 'returned')
    # check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Should fail because H0 has fields not in H
    H0 = np.zeros(3, dtype=sim_specs['out'] + gen_specs['out'] + alloc_specs['out'] + [('bad_name', bool), ('bad_name2', bool)])
    try:
        check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    except AssertionError:
        assert 1
    else:
        assert 0

    # Other individual tests
    check_inputs(libE_specs)
    check_inputs(alloc_specs)
    check_inputs(sim_specs)
    check_inputs(gen_specs)
    check_inputs(exit_criteria=exit_criteria, sim_specs=sim_specs, gen_specs=gen_specs)


def rmfield(a, *fieldnames_to_remove):
    return a[[name for name in a.dtype.names if name not in fieldnames_to_remove]]


if __name__ == "__main__":
    test_manager_exception()
    test_exception_raising_manager()
    test_proc_not_in_communicator()
    test_checking_inputs()
