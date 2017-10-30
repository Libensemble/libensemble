import sys, time, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs')) 
import aposmm_logic as al

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 
import libE_manager as man

from test_manager_main import make_criteria_and_specs_0

alloc = {'worker_ranks':set([1,2]),'persist_gen_ranks':set([])}

def test_failing_localopt_method():
    sim_specs_0, gen_specs_0, exit_criteria_0 = make_criteria_and_specs_0()
    H, _, _, _, _ = man.initialize(sim_specs_0, gen_specs_0, alloc, exit_criteria_0,[]) 
    H['returned'] = 1

    gen_specs_0['localopt_method'] = 'BADNAME'
    
    try: 
        al.advance_localopt_method(H, gen_specs_0, 0, 0, {'run_order': {0:[0,1]}})
    except: 
        assert 1, "Failed like it should have"
    else:
        assert 0, "Didn't fail like it should have"


def test_exception_raising():
    sim_specs_0, gen_specs_0, exit_criteria_0 = make_criteria_and_specs_0()
    H, _, _, _, _ = man.initialize(sim_specs_0, gen_specs_0, alloc, exit_criteria_0,[]) 
    H['returned'] = 1

    for method in ['LN_SBPLX','pounders']:
        gen_specs_0['localopt_method'] = method
        try: 
            al.advance_localopt_method(H, gen_specs_0,  0, 0, {'run_order': {0:[0,1]}})
        except: 
            assert 1, "Failed like it should have"
        else:
            assert 0, "Failed like it should have"


def test_decide_where_to_start_localopt():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../regression_tests'))

    from test_branin_aposmm import gen_out 
    H = np.zeros(10,dtype=gen_out + [('f',float),('returned',bool)])
    H['x'] = np.random.uniform(0,1,(10,2))
    H['f'] = np.random.uniform(0,1,10)
    H['returned'] = 1

    b = al.decide_where_to_start_localopt(H, 9, 1)
    assert len(b)==0

    b = al.decide_where_to_start_localopt(H, 9, 1, nu=0.01)
    assert len(b)==0


def test_calc_rk():
    rk = al.calc_rk(2,10,1)

    rk = al.calc_rk(2,10,1,10)
    assert np.isinf(rk)


def test_initialize_APOSMM():
    sim_specs_0, gen_specs_0, exit_criteria_0 = make_criteria_and_specs_0()
    H, _, _, _, _ = man.initialize(sim_specs_0, gen_specs_0, alloc, exit_criteria_0,[]) 

    al.initialize_APOSMM(H,gen_specs_0)
    

def test_queue_update_function():

    gen_specs_0 = {}
    gen_specs_0 = {}
    gen_specs_0['stop_on_NaNs'] = True
    gen_specs_0['combine_component_func'] = np.linalg.norm
    H = np.zeros(10, dtype=[('f_i',float),('returned',bool),('pt_id',int),('sim_id',int),('paused',bool)])

    H['sim_id'] = np.arange(0,10)
    H['pt_id'] = np.sort(np.concatenate([np.arange(0,5),np.arange(0,5)]))

    H['returned'][0:10:2] = 1 # All of the first components have been evaluated
    H['returned'][1] = 1 

    H['f_i'][4] = np.nan

    H,_ = al.queue_update_function(H, gen_specs_0,{})
    assert np.all(H['paused'][4:6])

    gen_specs_0['stop_partial_fvec_eval'] = True
    H['f_i'][6:10:2] = 0.5
    H,_ = al.queue_update_function(H, gen_specs_0,{})
    assert np.all(H['paused'][4:])


# if __name__ == "__main__":
#     import ipdb; ipdb.set_trace()
#     test_failing_localopt_method()
