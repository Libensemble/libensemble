import sys, time, os
import numpy as np
import numpy.lib.recfunctions

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 


import libensemble.libE_manager as man
import libensemble.tests.unit_tests.setup as setup

al = {'out':[]}
libE_specs ={'comm':{},'worker_ranks':set([1,2])}


def test_update_history_x_out():
    assert True


def test_termination_test():
    # termination_test should be True when we want to stop

    sim_specs_0, gen_specs_0, exit_criteria_0 = setup.make_criteria_and_specs_0()
    H, H_ind, term_test,_,_ = man.initialize(sim_specs_0, gen_specs_0, al, exit_criteria_0,[],libE_specs) 
    assert not term_test(H, H_ind)



    # Shouldn't terminate
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_1()
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[],libE_specs) 
    assert not term_test(H, H_ind)
    # 


    # Terminate because we've found a good 'g' value
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[],libE_specs) 
    H['g'][0] = -1
    H_ind = 1
    assert term_test(H, H_ind)
    # 

    
    # Terminate because everything has been given.
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[],libE_specs) 
    H['given'] = np.ones
    assert term_test(H, H_ind)
    # 
    

    # Terminate because enough time has passed
    H0 = np.zeros(3,dtype=sim_specs['out'] + gen_specs['out'])
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H0,libE_specs) 
    H_ind = 4
    H['given_time'][0] = time.time()
    time.sleep(0.5)
    assert term_test(H, H_ind)
    # 


def test_update_history_x_in():

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_1()
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[],libE_specs) 

    # Don't do anything when O is empty
    O = np.zeros(0, dtype=gen_specs['out'])

    H, H_ind = man.update_history_x_in(H, H_ind, 1, O)

    assert H_ind == 0


if __name__ == "__main__":
    test_update_history_x_in()
