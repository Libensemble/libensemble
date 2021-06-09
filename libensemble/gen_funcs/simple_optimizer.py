import numpy as np
import numpy.linalg as la
import scipy.optimize as sciopt

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

ct = 0

class Error(Exception):
    pass

class StopTagError(Error):
    # TODO: Can we remove this
    def __init__(self):
        pass

def double_extend(arr):
    out = np.zeros(len(arr)*2, dtype=type(arr[0]))
    out[0::2] = 2*np.array(arr)
    out[1::2] = 2*np.array(arr)+1
    return out

def simple_optimize(H, persis_info, gen_specs, libE_info):
    """ Uses scipy.optimize to solve objective function
    """
    f_i_idxs = persis_info.get('f_i_idxs')
    x_i_idxs = double_extend(f_i_idxs)
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(lb)

    def _f(x):
        global ct
        f_out, new_ct = _req_sims(x, f_i_idxs, x_i_idxs, ct, gen_specs, libE_info, get_f=True)
        ct = new_ct
        return f_out

    def _df(x):
        global ct
        df_out, new_ct = _req_sims(x, f_i_idxs, x_i_idxs, ct, gen_specs, libE_info, get_f=False)
        ct = new_ct
        return df_out

    while 1:
        # we need kron since we want @l multiples
        x0 = persis_info['rand_stream'].uniform(lb, ub, size=(n,))
        # TODO: prevent solutions that are close to previous solutions

        try:
            res = sciopt.minimize(_f, x0, jac=_df, method="BFGS", tol=1e-8)

            x = res.x
            convg = res.success

            if convg:
                print("===========================", flush=True)
                print("# FINISH \n\n", flush=True)
                print("# x={} ".format(x), flush=True)
                print("===========================", flush=True)
                return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        except StopTagError:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

def _req_sims(x, f_i_idxs, x_i_idxs, ct, gen_specs, libE_info, get_f):
    """ Request evaluation of function or its gradient (@get_f)

    Parmaters
    ---------
    x : np.ndarray(dtype=int)
        - input to evaluaate at
    x_i_idxs : np.ndarray(dtype=int)
        - which x_i_indices we are responsible for
    ct : int
        - counter for gen (internal book keeping)
    get_f : bool
        - True to request function, otherwise get gradient

    Returns
    -------
      : float or np.ndarray(dtype=float)
        - output of sim work
    ct : int
        - updated counter

    """
    n = len(x_i_idxs)

    if get_f:
        H_o = np.zeros(len(f_i_idxs), dtype=gen_specs['out'])
        H_o['obj_component'] = f_i_idxs
    else:
        H_o = np.zeros(len(x_i_idxs), dtype=gen_specs['out'])
        H_o['obj_component'] = x_i_idxs

    H_o['x'][:] = x
    H_o['pt_id'][:] = ct      
    H_o['num_sims_req'][:] = len(H_o)
    ct += 1

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

    if tag in [STOP_TAG, PERSIS_STOP]:
        raise StopTagError

    if get_f:
        f_is = calc_in['f_i'] # [f_i_idxs - np.max(f_i_idxs)]
        # print("[{}] sum(f)={:.2f} | x={}".format(libE_info['workerID'], np.sum(f_is), x), flush=True)
        return np.sum(f_is), ct
    else:
        gradf_is = np.zeros(len(x), dtype=float)
        gradf_is[x_i_idxs] = calc_in['gradf_i'][:]
        # print("[{}] grad(f)={} | x={}".format(libE_info['workerID'], gradf_is, x), flush=True)
        return gradf_is, ct
