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

if __name__ == "__main__":
    import ipdb; ipdb.set_trace() 

    test_exception_raising()
