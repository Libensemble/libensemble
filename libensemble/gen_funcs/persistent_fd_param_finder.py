import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.gen_funcs.support import sendrecv_mgr_worker_msg


def fd_param_finder(H, persis_info, gen_specs, libE_info):
    """
    This generation function loops through a set of suitable finite difference
    parameters for a mapping F from R^n to R^m.

    .. seealso::
        `test_persistent_fd_param_finder.py` <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_fd_param_finder.py>`_
    """
    x = gen_specs['user']['x']
    kmax = gen_specs['user']['kmax']
    n = len(x)

    comm = libE_info['comm']

    requested_points = np.zeros((kmax*n, n))

    h = 1  # Starting finite difference parameter

    # Send batches until a stop tag is received or we are happy with h
    O = np.zeros(kmax*n, dtype=gen_specs['out'])
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:

        ind = 0
        for i in range(n):
            for k in range(10):
                requested_points[ind] = x + (k+1)*h*np.eye(n)[i]
                ind += 1

        O['x'] = requested_points
        tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, O)

        # returned_values = calc_in['f']
        # print(returned_values)

        if h < 1e-1:
            tag = FINISHED_PERSISTENT_GEN_TAG
            break
        else:
            h = 0.5*h

    return O, persis_info, tag
