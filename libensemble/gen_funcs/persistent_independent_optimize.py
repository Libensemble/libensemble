import numpy as np
import scipy.optimize as sciopt

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.consensus_subroutines import (print_final_score, get_grad, get_func)


# TODO: Place this in support file or get rid fo
def double_extend(arr):
    """ Takes array [i_1,i_2,...i_k] and builds an extended array
        [2i_1, 2i_1+1, 2i_2, 2i_2+1, ..., 2i_k, 2i_k+1]

        For instances, given an array [0,1,2], we return
        [0,1, 2,3, 4,5]. This is useful for distributed sum of convex
        functions f_i which depend on two contiguous components x
        but not on the rest.
    """
    out = np.zeros(len(arr)*2, dtype=type(arr[0]))
    out[0::2] = 2*np.array(arr)
    out[1::2] = 2*np.array(arr)+1
    return out


def independent_optimize(H, persis_info, gen_specs, libE_info):
    """ Uses scipy.optimize to solve objective function
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']

    eps = persis_info['params']['eps']
    f_i_idxs = persis_info['f_i_idxs']

    def _f(x):
        return get_func(x, f_i_idxs, gen_specs, libE_info)

    def _df(x):
        return get_grad(x, f_i_idxs, gen_specs, libE_info)

    while 1:
        x0 = persis_info['rand_stream'].uniform(low=lb, high=ub)

        res = sciopt.minimize(_f, x0, jac=_df, method="BFGS", tol=eps,
                              options={'gtol': eps, 'norm': np.inf, 'maxiter': None})
        print_final_score(res.x, f_i_idxs, gen_specs, libE_info)

        start_pt, end_pt = f_i_idxs[0], f_i_idxs[-1]
        print('[Worker {}]: x={}'.format(persis_info['worker_num'], res.x[2*start_pt:2*end_pt]), flush=True)
        """
        try:
           res = sciopt.minimize(_f, x0, jac=_df, method="BFGS", tol=eps,
                                  options={'gtol': eps, 'norm': np.inf, 'maxiter': None})
           print_final_score(res.x, f_i_idxs, gen_specs, libE_info)

        except:
           print('hi', flush=True)
           return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
        """

        return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
