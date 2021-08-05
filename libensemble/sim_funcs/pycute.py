import numpy as np
from libensemble.tools.pycute_interface import Blackbox

def pycute_eval(H, persis_info, sim_specs, _):

    m = persis_info['params']['m']

    batch = len(H['x'])
    O = np.zeros(batch, dtype=sim_specs['out'])

    bbox = Blackbox(k=m)
    bbox.setup_new_prob(seed_num=0)
    bbox.set_scale()

    for k, x in enumerate(H['x']):
        i = H['obj_component'][k]  # which f_i

        if H[k]['get_grad']:
            O['gradf_i'][k] = bbox.df_i(x,i)
        else:
            O['f_i'][k] = bbox.f_i(x,i)

    return O, persis_info
