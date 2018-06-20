import sys, time, os
import numpy as np
import numpy.lib.recfunctions

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/alloc_funcs'))

import libensemble.libE_manager as man
import libensemble.tests.unit_tests.setup as setup
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first

al = {'alloc_f': give_sim_work_first,'out':[]}
libE_specs = {'comm': {}, 'worker_ranks':set([1,2])}

def test_decide_work_and_resources():

    sim_specs, gen_specs, exit_criteria = setup.make_criteria_and_specs_1()

    H, H_ind, _, _, _ = man.initialize(sim_specs, gen_specs, al, exit_criteria,[],libE_specs) 


    # Don't give out work when idle is empty
    worker_sets = {}
    worker_sets['nonpersis_w'] = {1:set([1,2,3,4]),'waiting': set()}
    Work, gen_info = al['alloc_f'](worker_sets, H, sim_specs, gen_specs, {})
    assert len(Work) == 0 
    # 
