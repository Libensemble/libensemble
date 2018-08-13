from __future__ import division
from __future__ import absolute_import

import numpy as np
from mpi4py import MPI
import sys

from libensemble.message_numbers import UNSET_TAG, STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG

def persistent_uniform(H,persis_info,gen_specs,libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs['gen_batch_size']`` uniformly sampled points.

    :See:
        ``libensemble/libensemble/tests/regression_tests/test_6-hump_camel_persistent_uniform_sampling.py``
    """
    ub = gen_specs['ub']
    lb = gen_specs['lb']
    n = len(lb)
    b = gen_specs['gen_batch_size']

    def make_batch():
        O = np.zeros(b, dtype=gen_specs['out'])
        for i in range(0,b):
            x = persis_info['rand_stream'].uniform(lb,ub,(1,n))
            O['x'][i] = x

        return {'calc_out':O,
                'libE_info': {'persistent':True},
                'calc_status': UNSET_TAG,
                'calc_type': EVAL_GEN_TAG
                }

    # Receive information from the manager (or a STOP_TAG)
    comm = libE_info['comm']
    status = MPI.Status()
    while True:
        comm.send(obj=make_batch(), dest=0, tag=EVAL_GEN_TAG)
        comm.probe(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() in [STOP_TAG, PERSIS_STOP]:
            break
        comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        comm.recv(buf=None, source=0)

    return O, persis_info, tag
