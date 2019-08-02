import os
import numpy as np
import pytest
import mock

from libensemble.libE import check_inputs, libE
import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from mpi4py import MPI


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
fname_abort = 'libE_history_at_abort_0.npy'


def test_manager_exception():

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0()

    try:
        os.remove(fname_abort)
    except OSError:
        pass
    with mock.patch('libensemble.libE.manager_main') as managerMock:
        managerMock.side_effect = Exception
        with mock.patch('libensemble.libE.comms_abort') as abortMock:
            abortMock.side_effect = Exception
            with pytest.raises(Exception):
                # libE({'out': [('f', float)]}, {'out': [('x', float)]}, {'sim_max': 1}, libE_specs={'comm': MPI.COMM_WORLD})
                # libE({'out': [('f', float)]}, {'out': [('x', float)]}, {'sim_max': 1}, libE_specs={'comm': fake_mpi})
                libE(sim_specs, gen_specs, exit_criteria, libE_specs={'comm': fake_mpi})
                pytest.fail('Expected exception')
            # Check npy file dumped
            assert os.path.isfile(fname_abort), "History file not dumped"
            os.remove(fname_abort)


def test_exception_raising_manager():
    # Intentionally running without sim_specs['in'] to test exception raising (Fails)
    with mock.patch('libensemble.libE.comms_abort') as abortMock:
        abortMock.side_effect = Exception
        with pytest.raises(Exception):
            H, _, _ = libE({'out': [('f', float)]}, {'out': [('x', float)]}, {'sim_max': 1}, libE_specs={'comm': MPI.COMM_WORLD})
            pytest.fail('Expected exception')


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
    H0 = np.zeros(3, dtype=sim_specs['out'] + gen_specs['out'] + \
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
    test_checking_inputs()
