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
    H, Hs_ind, term_test,_,_ = man.initialize(sim_specs_0, gen_specs_0, al, exit_criteria_0,[]) 
    assert not term_test(H, Hs_ind)



    # Shouldn't terminate
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 
    assert not term_test(H, Hs_ind)
    # 


    # Terminate because we've found a good 'g' value
    H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 
    H['g'][0] = -1
    Hs_ind = 1
    assert term_test(H, Hs_ind)
    # 

    
    # Terminate because everything has been given.
    H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 
    H['given'] = np.ones
    assert term_test(H, Hs_ind)
    # 
    

    # Terminate because enough time has passed
    H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 
    Hs_ind = 4
    H['given_time'][0] = time.time()
    time.sleep(0.5)
    assert term_test(H, Hs_ind)
    # 


def test_update_history_x_in():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_1()
    H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 

    O = np.zeros(2*len(H), dtype=gen_specs['out'])
    print(len(O))

    Hs_ind = man.update_history_x_in(H, Hs_ind, O)

    # assert Hs_ind == len(H)


def test_initialize_history():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_0()
    H0, _, _, _, _ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[]) 

    # Should fail because H0 has points with 'return'==False
    try:
        H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H0) 
    except AssertionError:
        assert 1
    else:
        assert 0

    # Should not fail 
    H0['returned']=True
    H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H0) 

    # Removing 'returned' and then testing again.
    H0 = rmfield( H0, 'returned')
    H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H0) 


    # Adding 'obj_component' but more than expected
    H1 = np.zeros(len(H0),dtype=[('obj_component',int)])
    H1['obj_component'] = np.arange(len(H1))
    H2 = np.lib.recfunctions.merge_arrays((H0,H1), flatten = True, usemask = False)
    gen_specs['params']['components'] = 2
    gen_specs['out'] += [('obj_component','int')]

    try: 
        H, Hs_ind,term_test,_,_ = man.initialize(sim_specs, gen_specs, al, exit_criteria,H2) 
    except AssertionError:
        assert 1
    else:
        assert 0

def rmfield( a, *fieldnames_to_remove ):
        return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]


if __name__ == "__main__":
    test_initialize_history()
