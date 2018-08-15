from libensemble.history import History
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
import libensemble.tests.unit_tests.setup as setup
from libensemble.message_numbers import WORKER_DONE
import numpy as np
from numpy import inf

# Comparing hist produced: options
# - hardcode array compare
# - compare from npy file - stored
# - compare selected values

#wrs_sub = np.array([(False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                    #(False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                    #(False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf)],
      #dtype=[('local_pt', '?'), ('priority', '<f8'), ('gen_worker', '<i8'), ('x_on_cube', '<f8'), ('paused', '?'), ('sim_id', '<i8'), ('given', '?'), ('sim_worker', '<i8'), ('returned', '?'), ('fvec', '<f8', (3,)), ('allocated', '?'), ('f', '<f8'), ('given_time', '<f8')])

wrs = np.array([(False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf),
                (False, 0., 0, 0., False, -1, False, 0, False, [0., 0., 0.], False, 0., inf)],
      dtype=[('local_pt', '?'), ('priority', '<f8'), ('gen_worker', '<i8'), ('x_on_cube', '<f8'), ('paused', '?'), ('sim_id', '<i8'), ('given', '?'), ('sim_worker', '<i8'), ('returned', '?'), ('fvec', '<f8', (3,)), ('allocated', '?'), ('f', '<f8'), ('given_time', '<f8')])

wrs2 = np.array([(0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False),
                 (0, False, 0., 0, False, 0., False, -1, inf, 0., False)],
      dtype=[('gen_worker', '<i8'), ('returned', '?'), ('x', '<f8'), ('sim_worker', '<i8'), ('allocated', '?'), ('g', '<f8'), ('given', '?'), ('sim_id', '<i8'), ('given_time', '<f8'), ('priority', '<f8'), ('paused', '?')])

exp_x_in_setup2 = np.array([(0, 0, 2, False, 0., 4.17022005e-01, False, False, False, inf, 0.),
                            (0, 1, 3, False, 0., 7.20324493e-01, False, False, False, inf, 0.),
                            (0, 2, 3, False, 0., 1.14374817e-04, False, False, False, inf, 0.),
                            (0, 3, 3, False, 0., 3.02332573e-01, False, False, False, inf, 0.),
                            (0, 4, 3, False, 0., 1.46755891e-01, False, False, False, inf, 0.),
                            (0, 5, 3, False, 0., 9.23385948e-02, False, False, False, inf, 0.),
                            (0, 6, 3, False, 0., 1.86260211e-01, False, False, False, inf, 0.),
                            (0, 7, 3, False, 0., 3.45560727e-01, False, False, False, inf, 0.),
                            (0, 8, 3, False, 0., 3.96767474e-01, False, False, False, inf, 0.),
                            (0, 9, 3, False, 0., 5.38816734e-01, False, False, False, inf, 0.)],
                           dtype=[('sim_worker', '<i8'), ('sim_id', '<i8'), ('gen_worker', '<i8'), ('paused', '?'), ('priority', '<f8'), ('x', '<f8'), ('allocated', '?'), ('returned', '?'), ('given', '?'), ('given_time', '<f8'), ('g', '<f8')])

# Could use fixtures here, but then cant run directly.
def hist_setup1(sim_max=10):
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_0(simx=sim_max)
    alloc_specs = {'alloc_f': give_sim_work_first, 'out':[('allocated', bool)]} #default for libE
    H0=[]
    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    return hist, sim_specs, gen_specs, exit_criteria, alloc_specs

def hist_setup2():
    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_1()
    alloc_specs = {'alloc_f': give_sim_work_first, 'out':[('allocated', bool)]} #default for libE
    H0=[]
    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    return hist, sim_specs, gen_specs, exit_criteria, alloc_specs

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Tests ========================================================================================

def test_hist_init_1():
    hist,_,_,_,_  = hist_setup1()
    assert np.array_equal(hist.H, wrs), "Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 0
    assert hist.sim_count == 0


def test_hist_init_2():
    hist,_,_,_,_  = hist_setup2()
    assert np.array_equal(hist.H, wrs2), "Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 0
    assert hist.sim_count == 0


def test_update_history_x_in_Oempty():
    hist,sim_specs,gen_specs,_,_ = hist_setup2()
    O = np.zeros(0, dtype=gen_specs['out'])
    gen_worker = 1
    hist.update_history_x_in(gen_worker, O)
    assert np.array_equal(hist.H, wrs2), "H Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 0
    assert hist.sim_count == 0
  
  
#Update values after gen.
#This function should be extended to include ref. to calc_status in updating.
def test_update_history_x_in():
    hist, _, gen_specs,_,_  = hist_setup2()
    #import pdb; pdb.set_trace()
    #calc_in = hist.H[gen_specs['in']][0]
    
    np.random.seed(1)
    single_rand = gen_specs['gen_f']() # np.random.uniform()
    
    # Check seeded correctly going in
    assert isclose(single_rand, 0.417022004702574), "Random numbers not correct before function"
    
    size = 1
    gen_worker = 2
    O = np.zeros(size, dtype=gen_specs['out'])
    O['x'] = single_rand
    
    hist.update_history_x_in(gen_worker, O)
    assert isclose(single_rand, hist.H['x'][0])
    assert hist.given_count == 0
    assert hist.index == 1
    assert hist.sim_count == 0    

    size = 9
    gen_worker = 3
    O = np.zeros(size, dtype=gen_specs['out'])
    O['x'] = gen_specs['gen_f'](size=9)
    hist.update_history_x_in(gen_worker, O)    
    
    # Compare by column
    for field in exp_x_in_setup2.dtype.names:
        np.allclose(hist.H[field], exp_x_in_setup2[field])
    
    assert hist.given_count == 0
    assert hist.index == 10
    assert hist.sim_count == 0


def test_update_history_x_in_preset_sim_ids():
    pass


#Note - prob need more setup here - as points not been generated yet...
#Also this raises question whether should check point has been generated
#and/or hist.index in test_update_history_x_out - should use pre-set array as
#would be once generated in here - and preset index.
#Also - can you do non-consecutive points??????????????????????????????????????????
def test_update_history_x_out():
    hist,_,_,_,_  = hist_setup1()
    
    # Update one point
    hist.update_history_x_out(q_inds=0, sim_worker=2)
    
    # Check updated values for point and counts
    assert hist.H['given'][0] == True
    assert hist.H['sim_worker'][0] == 2
    assert hist.given_count == 1
    
    # Check some unchanged values for point and counts
    assert hist.index == 0
    assert hist.sim_count == 0
    hist.H['returned'][0] == False
    hist.H['allocated'][0] == False 
    hist.H['f'][0] == 0.0
    hist.H['sim_id'][0] == -1
   
    # Check the rest of H is unaffected
    assert np.array_equal(hist.H[1:10], wrs[1:10]), "H Array slice does not match expected"
    
    
    # Update two further consecutive points
    my_qinds = np.arange(1,3)
    hist.update_history_x_out(q_inds=my_qinds, sim_worker=3)

    # Check updated values for point and counts
    assert np.all(hist.H['given'][0:3]) # Include previous point
    assert np.all(hist.H['sim_worker'][my_qinds]==3)
    assert hist.given_count == 3
    
    
    # Update three further non-consecutive points
    my_qinds = np.array([4,7,9])
    hist.update_history_x_out(q_inds=my_qinds, sim_worker=4)
    
    #Try to avoid tautological testing - compare columns
    assert np.array_equal(hist.H['given'], np.array([ True, True, True, False, True, False, False, True, False, True]))
    assert np.array_equal(hist.H['sim_worker'], np.array([2, 3, 3, 0, 4, 0, 0, 4, 0, 4]))
    assert np.all(hist.H['returned'] == False) # Should still be unaffected.
    
    # Check counts
    assert hist.given_count == 6
    assert hist.index == 0 # In real case this would be ahead.....
    assert hist.sim_count == 0    


# pss
def test_update_history_f():
    hist, sim_specs,_,_,_  = hist_setup2()
    #import pdb; pdb.set_trace()
    #calc_in = hist.H[gen_specs['in']][0]
    
    size = 1
    sim_ids = 0 # First row to be filled
    sim_ids = np.atleast_1d(sim_ids)
    calc_out = np.zeros(size, dtype=sim_specs['out'])
    print(calc_out)
    a = np.arange(9) - 4
    calc_out['g'] = sim_specs['sim_f'](a) #np.linalg.norm
    exp_val1 = calc_out['g'][0]
    print(calc_out)
    D_recv = {'calc_out': calc_out,
              'persis_info': {},
              'libE_info': {'H_rows': sim_ids},
              'calc_status': WORKER_DONE,
              'calc_type': 2}
   
    hist.update_history_f(D_recv)
    assert isclose(exp_val1, hist.H['g'][0])
    assert np.all(hist.H['returned'][0:1])
    assert np.all(hist.H['returned'][1:10] == False) #Check the rest
    assert hist.sim_count == 1
    assert hist.given_count == 0 # In real case this would be ahead.....
    assert hist.index == 0 # In real case this would be ahead....
    
    #check
    print(hist.H)
    print(exp_val1)
    print(hist.H['g'][0])
 
   #Now add more values...


def test_grow_H():
    hist,_,_,_,_  = hist_setup1(3)
    new_rows = 7
    hist.grow_H(k = new_rows)
    assert np.array_equal(hist.H, wrs), "Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 0
    assert hist.sim_count == 0    
    

def test_trim_H():
    hist,_,_,_,_  = hist_setup1(13)
    hist.index = 10
    H = hist.trim_H()
    assert np.array_equal(H, wrs), "Array does not match expected"
    assert hist.given_count == 0
    assert hist.index == 10
    assert hist.sim_count == 0


if __name__ == "__main__":
    #test_hist_init_1()
    #test_hist_init_2()
    #test_update_history_x_in_Oempty()
    #test_update_history_x_in()
    #test_update_history_x_out()
    test_update_history_f() # Not implemented yet
    #test_grow_H()
    #test_trim_H()
    
