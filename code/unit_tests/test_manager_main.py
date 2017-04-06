import sys, time
import numpy as np
sys.path.append('../src/')

import libE_manager as man

def test_update_history_x_out():
    assert(True)

def make_criteria_and_specs_1():
    sim_specs={'f': [np.linalg.norm], 'in':['x'], 'out':[('g','float')], 'params':{}}
    gen_specs={'f': [np.random.uniform], 'in':[], 'out':[('x','float'),('priority','float')], 'params':{}}
    exit_criteria={'sim_eval_max':10, 'stop_val':('g',-1), 'elapsed_clock_time':0.5}

    return sim_specs, gen_specs, exit_criteria


def test_termination_test():
    # termination_test should be False when we want to stop

    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()


    # Shouldn't terminate
    H, H_ind = man.initiate_H(sim_specs, gen_specs, exit_criteria) 
    assert(man.termination_test(H, H_ind, exit_criteria))
    # 


    # Terminate because we've found a good 'g' value
    H, H_ind = man.initiate_H(sim_specs, gen_specs, exit_criteria) 
    H['g'][0] = -1
    H_ind = 1
    assert(not man.termination_test(H, H_ind, exit_criteria))
    # 

    
    # Terminate because enough H_ind large 
    H, H_ind = man.initiate_H(sim_specs, gen_specs, exit_criteria) 
    H_ind = 10
    assert(not man.termination_test(H, H_ind, exit_criteria))
    # 
    

    # Terminate because enough time has passed
    H, H_ind = man.initiate_H(sim_specs, gen_specs, exit_criteria) 
    H_ind = 4
    H['given_time'][0] = time.time()
    time.sleep(0.5)
    assert(not man.termination_test(H, H_ind, exit_criteria))
    # 


def test_decide_work_and_resources():

    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, H_ind = man.initiate_H(sim_specs, gen_specs, exit_criteria) 


    # Don't give out work when idle is empty
    active_w = set([1,2,3,4])
    idle_w = set([])
    Work, H_ind = man.decide_work_and_resources(active_w, idle_w, H, H_ind, sim_specs, gen_specs)
    assert( len(Work) == 0 )
    # 


    # Don't give more gen work than space in the history. 
    active_w = set([])
    idle_w = set([1,2,3,4])
    H_ind = len(H)-2
    H['pt_id'][:len(H)-2] = np.arange(len(H)-2)
    Work, H_ind = man.decide_work_and_resources(active_w, idle_w, H, H_ind, sim_specs, gen_specs)
    assert( len(Work) == 2)
    # 


def test_update_history_x_in():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, H_ind = man.initiate_H(sim_specs, gen_specs, exit_criteria) 

    O = np.zeros(2*len(H), dtype=gen_specs['out'])

    H_ind = man.update_history_x_in(H, H_ind, O)

    assert(H_ind == len(H))

    

