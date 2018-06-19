import sys, time, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 

from libE import * 
from test_manager_main import make_criteria_and_specs_0

al = {}
libE_specs = {'comm':[], 'manager_ranks':set([0]), 'worker_ranks':set([1,2])}

def test_nonworker_and_nonmanager_rank():

    # Intentionally making worker 0 not be a manager or worker rank
    libE({'out':[('f',float)]},{'out':[('x',float)]},{'sim_max':1},libE_specs={'comm': MPI.COMM_WORLD,'manager_ranks':set([1]), 'worker_ranks':set([1])})


def test_checking_inputs():

    # Don't take more points than there is space in history.
    sim_specs, gen_specs, exit_criteria = make_criteria_and_specs_0()
    
    H0 = np.zeros(3,dtype=sim_specs['out'] + gen_specs['out'] + [('returned',bool)])

    # Should fail because H0 has points with 'return'==False
    try:
        check_inputs(libE_specs,al, sim_specs, gen_specs, {}, exit_criteria,H0) 
    except AssertionError:
        assert 1
    else:
        assert 0

    # Should not fail 
    H0['returned']=True
    check_inputs(libE_specs, al, sim_specs, gen_specs, {}, exit_criteria,H0) 

    # Removing 'returned' and then testing again.
    H0 = rmfield( H0, 'returned')
    check_inputs(libE_specs,al, sim_specs, gen_specs, {}, exit_criteria,H0) 


def rmfield( a, *fieldnames_to_remove ):
        return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]

if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    test_nonworker_and_nonmanager_rank()
