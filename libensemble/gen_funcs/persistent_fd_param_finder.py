import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.gen_funcs.support import sendrecv_mgr_worker_msg

def build_H0(x_inds, f_inds, gen_specs):

    U = gen_specs['user']
    x0 = U['x0']
    noise_h_mat = U['noise_h_mat']

    # This function constructs H0 to contain points to be sent back to the
    # manager to be evaluated

    n = len(x0)
    E = np.eye(n)
    nf = U['nf']

    H0 = np.zeros(len(x_inds)*len(f_inds)*nf, dtype=gen_specs['out'])
    ind = 0
    for i in x_inds:
        for j in f_inds: 
            for k in range(nf+1):
                if k != nf/2+1:
                    H0['x'][ind] = x0 + (k-nf/2)*noise_h_mat[i,j]*E[i]
                    H0['x_ind'][ind] = i 
                    H0['f_ind'][ind] = j 
                    H0['n_ind'][ind] = k
                    ind += 1

    return H0

def fd_param_finder(H, persis_info, gen_specs, libE_info):
    """
    This generation function loops through a set of suitable finite difference
    parameters for a mapping F from R^n to R^m.

    .. seealso::
        `test_persistent_fd_param_finder.py` <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_fd_param_finder.py>`_
    """
    U = gen_specs['user']

    p = U['p']
    x0 = U['x0']
    nf = U['nf']
    noise_h_mat = U['noise_h_mat']
    maxnoiseits = U['maxnoiseits']

    n = len(x0)
    Fhist0 = np.zeros((n,p,nf+1))

    comm = libE_info['comm']
    tag = None

    # Request evaluations of the base point x0 at all p f_inds
    H0 = np.zeros(p, dtype=gen_specs['out'])
    for j in range(p):
        H0['x'][j] = x0
        H0['x_ind'][j] = -1  # Marking these to know they are the basepoint
        H0['f_ind'][j] = j
        H0['n_ind'][j] = nf/2

    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H0)

    for i in range(n):
        for j in range(p):
            Fhist0[i,j,nf//2] = calc_in['f_val'][calc_in['f_ind']==j]

    H0 = build_H0(range(n), range(p), gen_specs)

    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H0)

    # import ipdb; ipdb.set_trace()

    # Send nf points for each (x_ind, f_ind) pair 
    while tag not in [STOP_TAG, PERSIS_STOP]:
        x_inds = calc_in['x_inds']
        f_inds = calc_in['f_inds']

        # Update Fhist0 
        for i in x_inds:
            for j in f_inds: 
                for k in range(nf+1):
                    if k != nf/2:
                        Fhist0[i,j,k] = calc_in['f_val'][np.logical_and.reduce(calc_in['x_ind']== i, calc_in['f_ind']==j, calc_in['n_ind']==k)]
            

        H0 = build_H0(x_inds, f_inds, gen_specs)

        tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H0)

        # returned_values = calc_in['f']
        # print(returned_values)

        if h < 1e-1:
            tag = FINISHED_PERSISTENT_GEN_TAG
            break
        else:
            h = 0.5*h

    return H0, persis_info, tag
