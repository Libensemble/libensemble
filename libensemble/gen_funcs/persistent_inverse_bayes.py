import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP
from libensemble.gen_funcs.support import sendrecv_mgr_worker_msg


def persistent_updater_after_likelihood(H, persis_info, gen_specs, libE_info):
    """
    """
    ub = gen_specs['ub']
    lb = gen_specs['lb']
    n = len(lb)
    comm = libE_info['comm']
    subbatch_size = gen_specs['subbatch_size']
    num_subbatches = gen_specs['num_subbatches']

    # Receive information from the manager (or a STOP_TAG)
    batch = -1
    tag = None
    w = np.nan
    while tag not in [STOP_TAG, PERSIS_STOP]:
        batch += 1
        O = np.zeros(subbatch_size*num_subbatches, dtype=gen_specs['out'])
        if np.all(~np.isnan(w)):
            O['weight'] = w
        for j in range(num_subbatches):
            for i in range(subbatch_size):
                row = subbatch_size*j + i
                O['x'][row] = persis_info['rand_stream'].uniform(lb, ub, (1, n))
                O['subbatch'][row] = j
                O['batch'][row] = batch
                O['prior'][row] = np.random.randn()
                O['prop'][row] = np.random.randn()

        # Send data and get next assignment
        tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, O)
        if calc_in is not None:
            w = O['prior'] + calc_in['like'] - O['prop']

    return O, persis_info, tag
