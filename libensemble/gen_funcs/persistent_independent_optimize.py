import numpy as np
import numpy.linalg as la
import scipy.optimize as sciopt

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

class Error(Exception):
    pass

class StopTagError(Error):
    # TODO: Can we remove this
    def __init__(self):
        pass

# TODO: Place this in support file
def double_extend(arr):
    out = np.zeros(len(arr)*2, dtype=type(arr[0]))
    out[0::2] = 2*np.array(arr)
    out[1::2] = 2*np.array(arr)+1
    return out

def independent_optimize(H, persis_info, gen_specs, libE_info):
    """ Uses scipy.optimize to solve objective function
    """
    f_i_idxs = persis_info.get('f_i_idxs')
    x_i_idxs = double_extend(f_i_idxs)
    ub = gen_specs['user']['ub'][x_i_idxs]
    lb = gen_specs['user']['lb'][x_i_idxs]
    n_i = len(lb)
    metadata = {'ct': 0, 'num_f_evals': 0, 'num_gradf_evals': 0, 'last_grad_norm' : 0 }

    def _f(x):
        ct = metadata['ct']
        f_out, new_ct = _req_sims(x, f_i_idxs, x_i_idxs, ct, gen_specs, libE_info, get_f=True)
        metadata['ct'] = new_ct
        metadata['num_f_evals'] += 1
        return f_out

    def _df(x):
        ct = metadata['ct']
        df_out, new_ct = _req_sims(x, f_i_idxs, x_i_idxs, ct, gen_specs, libE_info, get_f=False)
        metadata['ct'] = new_ct
        metadata['num_gradf_evals'] += 1
        metadata['last_grad_norm'] = la.norm(df_out,2)
        return df_out

    while 1:

        # TODO: prevent solutions that are close to previous solutions
        x0 = persis_info['rand_stream'].uniform(lb, ub, size=(n_i,))
        gtol = 1e-02

        try:
            res = sciopt.minimize(_f, x0, jac=_df, method="BFGS", tol=1e-2, options={'gtol': gtol, 'norm': np.inf, 'maxiter': None})

        except StopTagError:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        if res.success:
            # first sent @x back to alloc to remember
            H_o = np.zeros(1, dtype=gen_specs['out'])

            H_o[0]['x'][x_i_idxs] = res.x
            H_o[0]['converged'] = True
            H_o[0]['num_f_evals'] = metadata['num_f_evals']
            H_o[0]['num_gradf_evals'] = metadata['num_gradf_evals']
            # H_o[0]['num_sims_req'] =  1
            H_o[0]['pt_id'] = metadata['ct']

            print('Last gradient norm: {:.6f}. Expected at most: {:.6f}'.format(metadata['last_grad_norm'], gtol), flush=True)

            # tell alloc we have found min. alloc will require basic work job
            sendrecv_mgr_worker_msg(libE_info['comm'], output=H_o)

            # print("===========================", flush=True)
            # print("# FINISH \n", flush=True)
            # print("# x={} ".format(res.x), flush=True)
            # print("===========================", flush=True)
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
    H_o = np.zeros(len(f_i_idxs), dtype=gen_specs['out'])
    H_o['obj_component'] = f_i_idxs
    H_o['x'][:, x_i_idxs] = np.tile(x, (len(f_i_idxs), 1)) # broadcasts
    H_o['pt_id'][:] = ct      
    H_o['get_grad'][:] = not get_f
    H_o['converged'][:] = False
    ct += 1

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

    if tag in [STOP_TAG, PERSIS_STOP]:
        raise StopTagError

    if get_f:
        f_is = calc_in['f_i'] 
        return np.sum(f_is), ct
    else:
        gradf_is = calc_in['gradf_i'][:, x_i_idxs]
        return np.sum(gradf_is, axis=0), ct
