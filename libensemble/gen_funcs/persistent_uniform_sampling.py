import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP
from libensemble.gen_funcs.support import sendrecv_mgr_worker_msg


def persistent_uniform(H, persis_info, gen_specs, libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs['gen_batch_size']`` uniformly sampled points.

    .. seealso::
        `test_6-hump_camel_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_6-hump_camel_persistent_uniform_sampling.py>`_
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(lb)
    b = gen_specs['user']['gen_batch_size']
    comm = libE_info['comm']

    # Send batches until manager sends stop tag
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        O = np.zeros(b, dtype=gen_specs['out'])
        O['x'] = persis_info['rand_stream'].uniform(lb, ub, (b, n))
        tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, O)

    return O, persis_info, tag
