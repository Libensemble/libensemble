import numpy as np

def avail_worker_ids(W, persistent=None):
    if persistent is None:
        return W['worker_id'][W['active'] == 0]
    elif persistent:
        return W['worker_id'][np.logical_and(W['active'] == 0,
                                             W['persis_state'] != 0)]
    else:
        return W['worker_id'][np.logical_and(W['active'] == 0,
                                             W['persis_state'] == 0)]
