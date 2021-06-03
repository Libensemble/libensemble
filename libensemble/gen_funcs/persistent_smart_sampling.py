import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

def persistent_smart(H, persis_info, gen_specs, libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs['gen_batch_size']`` uniformly sampled points the first time it
    is called. Afterwards, it returns the number of points given. This can be
    used in either a batch or asynchronous mode by adjusting the allocation
    function.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    m = gen_specs['user']['m']  
    n = len(lb)
    b = gen_specs['user']['gen_batch_size']
    
    # Send batches until manager sends stop tag
    tag = None
    ct = 0

    while tag not in [STOP_TAG, PERSIS_STOP]:

        H_o = np.zeros(b*m, dtype=gen_specs['out'])

        for i in range(b):
            x = persis_info['rand_stream'].uniform(lb, ub, (1, n))

            H_o['x'][i*m:(i+1)*m, :] = np.tile(x, (m, 1)) # duplicate `x` @m times
                                                          # TODO: If `x` is large, can we ref it
            H_o['pt_id'][i*m:(i+1)*m] = ct                # every @m evals is for a single x_i
            H_o['obj_component'][i*m:(i+1)*m] = np.arange(0,m)

            ct += 1

        print("sending ...")
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        print("received ...")

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
