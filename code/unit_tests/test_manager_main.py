import sys, time
sys.path.append('../src/')

import libE_manager

def test_update_history_x_out():
    assert(True)

def test_termination_test():
    # termination_test should be False when we want to stop

    exit_criteria={'sim_eval_max':10, 'stop_val':('g',-1), 'elapsed_clock_time':0.5}


    # Shouldn't terminate
    H, H_ind = libE_manager.initiate_H(sim_specs={'out':[('g','float')]}, gen_specs={'out':[('x','float')]}, exit_criteria=exit_criteria) 
    assert(libE_manager.termination_test(H, H_ind, exit_criteria))


    # Terminate because we've found a good 'g' value
    H, H_ind = libE_manager.initiate_H(sim_specs={'out':[('g','float')]}, gen_specs={'out':[('x','float')]}, exit_criteria=exit_criteria) 
    H['g'][0] = -1
    H_ind = 1
    assert(not libE_manager.termination_test(H, H_ind, exit_criteria))

    
    # Terminate because enough H_ind large 
    H, H_ind = libE_manager.initiate_H(sim_specs={'out':[('g','float')]}, gen_specs={'out':[('x','float')]}, exit_criteria=exit_criteria) 
    H_ind = 10
    assert(not libE_manager.termination_test(H, H_ind, exit_criteria))

    # Terminate because enough time has passed
    H, H_ind = libE_manager.initiate_H(sim_specs={'out':[('g','float')]}, gen_specs={'out':[('x','float')]}, exit_criteria=exit_criteria) 
    H_ind = 4
    H['given_time'][0] = time.time()
    time.sleep(0.5)

    assert(not libE_manager.termination_test(H, H_ind, exit_criteria))


