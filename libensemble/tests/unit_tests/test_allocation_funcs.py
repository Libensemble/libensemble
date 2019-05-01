from mpi4py import MPI

import libensemble.libE_manager as man
import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.history import History

al = {'alloc_f': give_sim_work_first, 'out': []}
libE_specs = {'comm': MPI.COMM_WORLD}
H0 = []


def test_decide_work_and_resources():

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_1()
    hist = History(al, sim_specs, gen_specs, exit_criteria, H0)

    mgr = man.Manager(hist, libE_specs, al, sim_specs, gen_specs, exit_criteria)
    W = mgr.W

    # Don't give out work when all workers are active
    W['active'] = 1
    Work, persis_info = al['alloc_f'](W, hist.H, sim_specs, gen_specs, al, {})
    assert len(Work) == 0


if __name__ == "__main__":
    test_decide_work_and_resources()
