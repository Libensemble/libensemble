import sys, time, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src')) 

from libE import * 

def test_nonworker_and_nonmanager_rank():

    # Intentionally making worker 0 not be a manager or worker rank
    libE({'out':[('f',float)]},{'out':[('x',float)]},{'sim_max':1},alloc_specs={'manager_ranks':set([1]), 'worker_ranks':set([1])})


if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    test_nonworker_and_nonmanager_rank()
