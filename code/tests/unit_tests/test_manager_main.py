import sys, time, os
import numpy as np
import numpy.lib.recfunctions

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 


import libE_manager as man

al = {'worker_ranks':set([1,2]),'persist_gen_ranks':set([])}

def test_update_history_x_out():
    assert True

def make_criteria_and_specs_0():
    sim_specs={'sim_f': [np.linalg.norm], 'in':['x_on_cube'], 'out':[('f',float),('fvec',float,3)], }
    gen_specs={'gen_f': [np.random.uniform], 'in':[], 'out':[('x_on_cube',float),('priority',float),('local_pt',bool)], 'ub':np.ones(1), 'nu':0}
    exit_criteria={'sim_max':10}

    return sim_specs, gen_specs, exit_criteria

def make_criteria_and_specs_1():
    sim_specs={'sim_f': [np.linalg.norm], 'in':['x'], 'out':[('g',float)], }
    gen_specs={'gen_f': [np.random.uniform], 'in':[], 'out':[('x',float),('priority',float)], }
    exit_criteria={'sim_max':10, 'stop_val':('g',-1), 'elapsed_wallclock_time':0.5}

    return sim_specs, gen_specs, exit_criteria


def test_termination_test():
    # termination_test should be True when we want to stop

    sim_specs_0, gen_specs_0, exit_criteria_0 = make_criteria_and_specs_0()
    H, H_ind, term_test,_,_ = man.initialize(sim_specs_0, gen_specs_0, al, exit_criteria_0,[]) 
    assert not term_test(H, H_ind)



    # Shouldn't terminate
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 
    assert not term_test(H, H_ind)
    # 


    # Terminate because we've found a good 'g' value
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 
    H['g'][0] = -1
    H_ind = 1
    assert term_test(H, H_ind)
    # 

    
    # Terminate because everything has been given.
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 
    H['given'] = np.ones
    assert term_test(H, H_ind)
    # 
    

    # Terminate because enough time has passed
    H0 = np.zeros(3,dtype=sim_specs['out'] + gen_specs['out'])
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H0) 
    H_ind = 4
    H['given_time'][0] = time.time()
    time.sleep(0.5)
    assert term_test(H, H_ind)
    # 


def test_update_history_x_in():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 

    O = np.zeros(2*len(H), dtype=gen_specs['out'])
    print(len(O))

    H_ind = man.update_history_x_in(H, H_ind, 1, O)

    # assert H_ind == len(H)


# if __name__ == "__main__":
