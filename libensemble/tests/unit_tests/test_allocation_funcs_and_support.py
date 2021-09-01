from mpi4py import MPI

import numpy as np
import libensemble.manager as man
import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.tools.alloc_support import AllocSupport
from libensemble.tools import add_unique_random_streams
from libensemble.history import History
from libensemble.tests.unit_tests.test_history import wrs_H0

al = {'alloc_f': give_sim_work_first, 'out': []}
libE_specs = {'mpi_comm': MPI.COMM_WORLD}
H0 = []

W = np.array([(1, 0, 0, 0, 0, False), (2, 0, 0, 0, 0, False),
            (3, 0, 0, 0, 0, False), (4, 0, 0, 0, 0, False)],
            dtype=[('worker_id', '<i8'), ('active', '<i8'), ('persis_state', '<i8'),
            ('worker_group', '<i8'), ('active_recv', '<i8'), ('zero_resource_worker', '?')])

H = np.array([(False, np.inf, False, 1, 0., 1., False, 1.6e09, 0, 1, False, [-0.49,  0.88], np.inf, 0., 0, False, False, [0., 0.], 1.6e09, np.inf),
                    (False, np.inf, False, 1, 0., 1., False, 1.6e09, 0, 1, False, [-2.99, -0.79], np.inf, 0., 1, False, False, [0., 0.], 1.6e09, np.inf),
                    (False, np.inf, False, 1, 0., 1., False, 1.6e09, 0, 1, False, [-2.11, -1.63], np.inf, 0., 2, False, False, [0., 0.], 1.6e09, np.inf),
                    (False, np.inf, False, 1, 0., 1., False, 1.6e09, 0, 1, False, [-1.88, -0.61], np.inf, 0., 3, False, False, [0., 0.], 1.6e09, np.inf),
                    (False, np.inf, False, 1, 0., 1., False, 1.6e09, 0, 1, False, [-0.61,  0.15], np.inf, 0., 4, False, False, [0., 0.], 1.6e09, np.inf)],
                    dtype=[('given', '?'), ('last_given_back_time', '<f8'), ('given_back', '?'), ('gen_worker', '<i8'), ('returned_time', '<f8'),
                    ('priority', '<f8'), ('kill_sent', '?'), ('gen_time', '<f8'), ('sim_worker', '<i8'), ('resource_sets', '<i8'), ('returned', '?'),
                    ('x', '<f8', (2,)), ('last_given_time', '<f8'), ('f', '<f8'), ('sim_id', '<i8'), ('cancel_requested', '?'), ('allocated', '?'),
                    ('x_on_cube', '<f8', (2,)), ('last_gen_time', '<f8'), ('given_time', '<f8')])

def test_decide_work_and_resources():

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_1()
    hist = History(al, sim_specs, gen_specs, exit_criteria, H0)

    mgr = man.Manager(hist, libE_specs, al, sim_specs, gen_specs, exit_criteria)
    W = mgr.W

    # Don't give out work when all workers are active
    W['active'] = 1
    Work, persis_info = al['alloc_f'](W, hist.H, sim_specs, gen_specs, al, {})
    assert len(Work) == 0

def test_allocsupport_init():
    als = AllocSupport(W, H)
    assert als.manage_resources, \
        "AllocSupport should be managing resources for sim_work and gen_work."


persis_info = add_unique_random_streams({}, 5)


if __name__ == "__main__":
    test_decide_work_and_resources()
    test_allocsupport_init()
