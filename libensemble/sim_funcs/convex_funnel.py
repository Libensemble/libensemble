import numpy as np

def EvaluateFunction(x, component=np.nan):
    f = np.log(1 + 10 * la.norm(x-1, 2)**2)
    return f


def EvaluateJacobian(x, component=np.nan):
    df = 10/(1 + la.norm(x-1,2)**2) * np.sign(x-1)
    return df

def convex_funnel_eval(H, persis_info, sim_specs, _):

    batch = len(H['x'])
    O = np.zeros(batch, dtype=sim_specs['out'])

    for i, x in enumerate(H['x']):
        obj_component = H['obj_component'][i]  # which f_i
        O['gradf_i'][i] = EvaluateJacobian(x, obj_component)

        # if H[i]['get_grad']:
        #     O['gradf_i'][i] = EvaluateJacobian(x, obj_component)
        # else:
        #     O['f_i'][i] = EvaluateFunction(x, obj_component)

    return O, persis_info
