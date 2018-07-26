#gen_func

from __future__ import division
from __future__ import absolute_import

import numpy as np
from mpi4py import MPI
import sys
import pdb

from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG

def persistent_updater_after_likelihood(H,persis_info,gen_specs,libE_info):
    """
    """
    ub = gen_specs['ub']
    lb = gen_specs['lb']
    n = len(lb)
    comm = libE_info['comm']

    # Receive information from the manager (or a STOP_TAG) 
    status = MPI.Status()

    batch = -1
    while 1: 
        batch += 1
        O = np.zeros(gen_specs['subbatch_size']*gen_specs['num_subbatches'], dtype=gen_specs['out'])
        row = -1
        for j in range(gen_specs['num_subbatches']):
            for i in range(0,gen_specs['subbatch_size']):
                row += 1 
                x = persis_info['rand_stream'].uniform(lb,ub,(1,n))
                O['x'][row] = x
                O['subbatch'][row] = j
                O['batch'][row] = batch
                O['prior'][row] = np.random.randn()
                O['prop'][row] = np.random.randn()
        
        # What is being sent to manager to pass on to workers
        D = {'calc_out':O,
             'libE_info': {'persistent':True},
             'calc_status': UNSET_TAG,
             'calc_type': EVAL_GEN_TAG
            }
        
        # Sending data
        comm.send(obj=D,dest=0,tag=EVAL_GEN_TAG)
        
        # Not sure what probe is doing, possibly bothering manager to see if its quiting time
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)          
        tag = status.Get_tag()
        if tag in [STOP_TAG, PERSIS_STOP]:   
            break 
        else: 
            Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        
        # Not sure why there are two comm.recv
        libE_info = Work['libE_info']
        calc_in = comm.recv(buf=None, source=0)
        O['weight'] = O['prior'] + calc_in['like'] - O['prop']
        

    return O, persis_info, tag
