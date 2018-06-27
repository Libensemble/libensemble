from __future__ import division
from __future__ import absolute_import

import numpy as np
from mpi4py import MPI
import sys

from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG

import nlopt

def persistent_uniform(H,gen_info,gen_specs,libE_info):
    """
    This generator
        - Returns "gen_batch_size" uniformly sampled points when called in
          nonpersistent mode. 
        - Performs a persistent nlopt local optimization run when called in
          persistent mode.
    """
    ub = gen_specs['ub']
    lb = gen_specs['lb']
    n = len(lb)
    b = gen_specs['gen_batch_size']
    comm = libE_info['comm']

    # Receive information from the manager (or a STOP_TAG) 
    status = MPI.Status()

    while 1: 
        O = np.zeros(b, dtype=gen_specs['out'])
        for i in range(0,b):
            x = gen_info['rand_stream'].uniform(lb,ub,(1,n))
            O['x'][i] = x

        D = {'calc_out':O,
             'libE_info': {'persistent':True},
             'calc_status': UNSET_TAG,
             'calc_type': EVAL_GEN_TAG
            }
        
        comm.send(obj=D,dest=0,tag=EVAL_GEN_TAG)

        #libE_info = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)          
        tag = status.Get_tag()
        if tag in [STOP_TAG, PERSIS_STOP]:   
            break 
        else:
            Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
            
        #_ = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        #calc_in = comm.recv(buf=None,source=0,tag=MPI.ANY_TAG,status=status)
        libE_info = Work['libE_info']
        calc_in = comm.recv(buf=None, source=0)

    return O, gen_info, tag
