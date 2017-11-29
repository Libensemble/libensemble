from __future__ import division
from __future__ import absolute_import

import numpy as np
from mpi4py import MPI

def uniform_random_sample_with_different_nodes_and_ranks(H,gen_info,gen_specs,libE_info):
    """
    Generates points uniformly over the domain defined by gen_specs['ub'] and
    gen_specs['lb']. Also randomly requests a different number of nodes to be
    used in the evaluation of the generated point.
    """

    del libE_info # Ignored parameter

    ub = gen_specs['ub']
    lb = gen_specs['lb']
    n = len(lb)

    if len(H) == 0: 
        b = gen_specs['initial_batch_size']

        O = np.zeros(b, dtype=gen_specs['out'])
        for i in range(0,b):
            # x = np.random.uniform(lb,ub,(1,n))
            x = gen_info['rand_stream'][MPI.COMM_WORLD.Get_rank()].uniform(lb,ub,(1,n))
            O['x'][i] = x
            O['num_nodes'][i] = 1
            O['ranks_per_node'][i] = 16
            O['priority'] = 1
        
    else:
        O = np.zeros(1, dtype=gen_specs['out'])
        O['x'] = len(H)*np.ones(n)
        O['num_nodes'] = np.random.randint(1,gen_specs['max_num_nodes']+1) 
        O['ranks_per_node'] = np.random.randint(1,gen_specs['max_ranks_per_node']+1)
        O['priority'] = 10*O['num_nodes']

    return O, gen_info


def uniform_random_sample_obj_components(H,gen_info,gen_specs,libE_info):
    """
    Generates points uniformly over the domain defined by gen_specs['ub'] and
    gen_specs['lb'] but requests each component be evaluated separately.
    """
    del libE_info # Ignored parameter

    ub = gen_specs['ub']
    lb = gen_specs['lb']

    n = len(lb)
    m = gen_specs['components']
    b = gen_specs['gen_batch_size']

    O = np.zeros(b*m, dtype=gen_specs['out'])
    for i in range(0,b):
        # x = np.random.uniform(lb,ub,(1,n))
        x = gen_info['rand_stream'][MPI.COMM_WORLD.Get_rank()].uniform(lb,ub,(1,n))

        O['x'][i*m:(i+1)*m,:] = np.tile(x,(m,1))
        # O['priority'][i*m:(i+1)*m] = np.random.uniform(0,1,m)
        O['priority'][i*m:(i+1)*m] = gen_info['rand_stream'][MPI.COMM_WORLD.Get_rank()].uniform(0,1,m)
        O['obj_component'][i*m:(i+1)*m] = np.arange(0,m)

        O['pt_id'][i*m:(i+1)*m] = len(H)//m+i

    return O, gen_info

def uniform_random_sample(H,gen_info,gen_specs,libE_info):
    """
    Generates points uniformly over the domain defined by gen_specs['ub'] and
    gen_specs['lb'].
    """
    del libE_info # Ignored parameter

    ub = gen_specs['ub']
    lb = gen_specs['lb']

    n = len(lb)
    b = gen_specs['gen_batch_size']

    O = np.zeros(b, dtype=gen_specs['out'])
    for i in range(0,b):
        # x = np.random.uniform(lb,ub,(1,n))
        x = gen_info['rand_stream'][MPI.COMM_WORLD.Get_rank()].uniform(lb,ub,(1,n))

        O['x'][i] = x

    return O, gen_info
