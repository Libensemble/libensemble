import sys, time, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs')) 
import aposmm_logic as al

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 
import libE_manager as man

from test_manager_main import make_criteria_and_specs_0

def test_failing_localopt_method():
    sim_specs_0, gen_specs_0, exit_criteria_0 = make_criteria_and_specs_0()
    H, H_ind, term_test = man.initialize(sim_specs_0, gen_specs_0, exit_criteria_0,[]) 

    gen_specs_0['params']['localopt_method'] = 'BADNAME'
    
    try: 
        al.advance_localopt_method(H, gen_specs_0['params'], np.array([0,1]), 0)
    except: 
        assert 1, "Failed like it should have"
    else:
        assert 0, "Didn't fail like it should have"


def test_exception_raising():
    sim_specs_0, gen_specs_0, exit_criteria_0 = make_criteria_and_specs_0()
    H, H_ind, term_test = man.initialize(sim_specs_0, gen_specs_0, exit_criteria_0,[]) 

    for method in ['LN_SBPLX','pounders']:
        gen_specs_0['params']['localopt_method'] = method
        try: 
            al.advance_localopt_method(H, gen_specs_0['params'], np.array([0,1]), 0)
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



# if __name__ == "__main__":
