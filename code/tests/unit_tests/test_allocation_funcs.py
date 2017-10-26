import sys, time, os
import numpy as np
import numpy.lib.recfunctions

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/alloc_funcs'))

import libE_manager as man
from test_manager_main import make_criteria_and_specs_1
from give_sim_work_first import give_sim_work_first

al = {'alloc_f': give_sim_work_first, 'worker_ranks':set([1,2]),'persist_gen_ranks':set([])}

def test_decide_work_and_resources():

    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 


    # Don't give out work when idle is empty
    active_w = set([1,2,3,4])
    idle_w = set()
    Work, gen_info = al['alloc_f'](active_w, idle_w, H, Hs_ind, sim_specs, gen_specs, term_test, {})
    assert len(Work) == 0 
    # 

if __name__ == "__main__":
    test_initialize_history()
