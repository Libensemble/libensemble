import sys, time, os
import numpy as np
import numpy.lib.recfunctions

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 


import libE_manager as man

al = {'worker_ranks':set([1,2]),'persist_gen_ranks':set([])}

def test_update_history_x_out():
    assert True

def make_criteria_and_specs_0():
    sim_specs={'sim_f': [np.linalg.norm], 'in':['x_on_cube'], 'out':[('f',float),('fvec',float,3)], 'params':{}}
    gen_specs={'gen_f': [np.random.uniform], 'in':[], 'out':[('x_on_cube',float),('priority',float)], 'params':{}}
    exit_criteria={'sim_max':10}

    return sim_specs, gen_specs, exit_criteria

def make_criteria_and_specs_1():
    sim_specs={'sim_f': [np.linalg.norm], 'in':['x'], 'out':[('g',float)], 'params':{}}
    gen_specs={'gen_f': [np.random.uniform], 'in':[], 'out':[('x',float),('priority',float)], 'params':{}}
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
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 
    H_ind = 4
    H['given_time'][0] = time.time()
    time.sleep(0.5)
    assert term_test(H, H_ind)
    # 


def test_decide_work_and_resources():

    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 


    # Don't give out work when idle is empty
    active_w = set([1,2,3,4])
    idle_w = set()
    Work = man.decide_work_and_resources(active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test)
    assert len(Work) == 0 
    # 


    # # Don't give more gen work than space in the history. 
    # active_w = set()
    # idle_w = set([1,2,3,4])
    # H_ind = len(H)-2 # Only want one more Gen Point
    # H['sim_id'][:len(H)-2] = np.arange(len(H)-2)
    # Work = man.decide_work_and_resources(active_w, idle_w, H, H_ind, sim_specs, gen_specs)
    # print(len(Work))
    # assert len(Work) == 2 # Length 2 because first work element is all points to be evaluated, second element is gen point
    # # 


def test_update_history_x_in():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 

    O = np.zeros(2*len(H), dtype=gen_specs['out'])
    print(len(O))

    H_ind = man.update_history_x_in(H, H_ind, O)

    # assert H_ind == len(H)


def test_initialize_history():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_0()
    H0, _, _, _, _ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 

    # Should fail because H0 has points with 'return'==False
    try:
        H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H0) 
    except AssertionError:
        assert 1
    else:
        assert 0

    # Should not fail 
    H0['returned']=True
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H0) 

    # Removing 'returned' and then testing again.
    H0 = rmfield( H0, 'returned')
    H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H0) 


    # Adding 'obj_component' but more than expected
    H1 = np.zeros(len(H0),dtype=[('obj_component',int)])
    H1['obj_component'] = np.arange(len(H1))
    H2 = np.lib.recfunctions.merge_arrays((H0,H1), flatten = True, usemask = False)
    gen_specs['params']['components'] = 2
    gen_specs['out'] += [('obj_component','int')]

    try: 
        H, H_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H2) 
    except AssertionError:
        assert 1
    else:
        assert 0

def rmfield( a, *fieldnames_to_remove ):
        return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]


if __name__ == "__main__":
    test_initialize_history()
