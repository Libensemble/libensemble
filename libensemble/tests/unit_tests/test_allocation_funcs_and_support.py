from mpi4py import MPI

import numpy as np
import libensemble.manager as man
import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, AllocException
from libensemble.tools import add_unique_random_streams
from libensemble.history import History
from libensemble.resources.scheduler import ResourceScheduler
from libensemble.resources.resources import Resources
from libensemble.tools import add_unique_random_streams

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

def initialize_resources():
    Resources.init_resources({'comms': 'local', 'nworkers': 4, 'num_resource_sets': 4})
    Resources.resources.set_resource_manager(4)

def clear_resources():
    Resources.resources = None

def test_decide_work_and_resources():

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_1()
    hist = History(al, sim_specs, gen_specs, exit_criteria, H0)

    mgr = man.Manager(hist, libE_specs, al, sim_specs, gen_specs, exit_criteria)
    W = mgr.W

    # Don't give out work when all workers are active
    W['active'] = 1
    Work, persis_info = al['alloc_f'](W, hist.H, sim_specs, gen_specs, al, {})
    assert len(Work) == 0

def test_als_init_normal():

    als = AllocSupport(W, H)
    assert als.manage_resources, \
        "AllocSupport instance should be managing resources for sim_work and gen_work."

    assert not als.resources, \
        "AllocSupport instance shouldn\'t be assigned a Resources object if Resources " + \
        "has not been initialized."

def test_als_init_withresources():
    initialize_resources()

    als = AllocSupport(W, H)
    assert isinstance(als.resources, Resources), \
        "AllocSupport instance didn\'t get assigned a Resources object."
    assert isinstance(als.sched, ResourceScheduler), \
        "AllocSupport instance didn\'t get assigned a ResourceScheduler object."

    clear_resources()

def test_als_assign_resources():

    als = AllocSupport(W, H)
    assert not als.assign_resources(4), \
        "AllocSupport instance shouldn\'t assign resources if not assigned a Scheduler"

    initialize_resources()
    als = AllocSupport(W, H)
    assert len(als.assign_resources(4)), \
        "AllocSupport didn\'t return a resource set team."

    clear_resources()

def test_als_worker_ids():
    als = AllocSupport(W, H)
    assert als.avail_worker_ids() == [1, 2, 3 ,4], \
        "avail_worker_ids() didn\'t return expected available worker list."

    W_ps = W.copy()
    W_ps['persis_state'] = np.array([2, 0 ,0, 0])
    als = AllocSupport(W_ps, H)
    assert als.avail_worker_ids(persistent=2) == [1], \
        "avail_worker_ids() didn\'t return expected persistent worker list."

    W_ar = W.copy()
    W_ar['active_recv'] = np.array([1, 0 ,0, 0])
    W_ar['persis_state'] = np.array([2, 0 ,0, 0])
    als = AllocSupport(W_ar, H)
    assert als.avail_worker_ids(persistent=2, active_recv=True) == [1], \
        "avail_worker_ids() didn\'t return expected persistent worker list."

    flag = 1
    try:
        als.avail_worker_ids(active_recv=True)
    except AllocException:
        flag = 0
    assert flag == 0, "AllocSupport didn't error on invalid options for avail_worker_ids()"

    W_ar = W.copy()
    W_ar['active_recv'] = np.array([1, 0 ,0, 0])
    W_ar['persis_state'] = np.array([2, 0 ,0, 0])
    als = AllocSupport(W_ar, H)
    assert als.avail_worker_ids(persistent=EVAL_GEN_TAG, active_recv=True) == [1], \
        "avail_worker_ids() didn\'t return expected persistent worker list."

    W_zrw = W.copy()
    W_zrw['zero_resource_worker'] = np.array([1, 0 ,0, 0])
    als = AllocSupport(W_zrw, H)
    assert als.avail_worker_ids(zero_resource_workers=True) == [1], \
        "avail_worker_ids() didn\'t return expected zero resource worker list."

def test_als_evaluate_gens():
    W_gens = W.copy()
    W_gens['active'] = np.array([2, 0 ,2, 0])
    als = AllocSupport(W_gens, H)
    assert als.count_gens() == 2, \
        "count_gens() didn't return correct number of active generators"

    assert als.test_any_gen(), \
        "test_any_gen() didn't return True on a generator worker being active."

    W_gens['persis_state'] = np.array([2, 0 ,0, 0])

    assert als.count_persis_gens() == 1, \
        "count_persis_gens() didn't return correct number of active persistent generators"

def test_als_sim_work():
    persis_info = add_unique_random_streams({}, 5)
    als = AllocSupport(W, H)
    Work = {}
    als.sim_work(Work, 1, ['x'], np.array([0, 1, 2, 3, 4]), persis_info[1])
    assert Work[1]['H_fields'] == ['x'], \
        "H_fields were not assigned to Work dict correctly."

    assert Work[1]['persis_info'] == persis_info[1], \
        "persis_info was not assigned to Work dict correctly."

    assert Work[1]['tag'] == EVAL_SIM_TAG, \
        "sim_work didn't assign tag to EVAL_SIM_TAG"

    assert not Work[1]['libE_info']['rset_team'], \
        "rset_team should not be defined if Resources hasn\'t been initialized?"

    assert all([i in Work[1]['libE_info']['H_rows'] for i in np.array([0, 1, 2, 3, 4])]), \
        "H_rows weren't assigned to libE_info correctly."

    W_ps = W.copy()
    W_ps['persis_state'] = np.array([1, 0 ,0, 0])
    als = AllocSupport(W_ps, H)
    Work = {}
    als.sim_work(Work, 1, ['x'], np.array([0, 1, 2, 3, 4]), persis_info[1], persistent=True)

    assert not len(Work[1]['libE_info']['rset_team']), \
        "Resource set should be empty for persistent workers."

    initialize_resources()
    als = AllocSupport(W, H)
    Work = {}
    als.sim_work(Work, 1, ['x'], np.array([0, 1, 2, 3, 4]), persis_info[1])

    assert len(Work[1]['libE_info']['rset_team']), \
        "Resource set should be assigned in libE_info"

    clear_resources()

if __name__ == "__main__":
    test_decide_work_and_resources()

    test_als_init_normal()
    test_als_init_withresources()
    test_als_assign_resources()
    test_als_worker_ids()
    test_als_evaluate_gens()
    test_als_sim_work()
