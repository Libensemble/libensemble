import numpy as np

const = 100

def EvaluateFunction(x, component=np.nan):
    """
    Evaluates the chained Rosenbrock function
    """

    if np.isnan(component):
        assert False
    else:
        i = component
        x1 = x[i]
        x2 = x[i+1]
        f = 100 * (x1**2 - x2)**2 + (x1-1)**2

    return 1.0/const * f


def EvaluateJacobian(x, component=np.nan):
    """
    Evaluates the chained Rosenbrock Jacobian
    """

    df = np.zeros(len(x), dtype=float)

    if np.isnan(component):
        assert False

    else:
        i = component
        x1 = x[i]
        x2 = x[i+1]

        df[i] = 400 * x1 * (x1**2 - x2) + 2 * (x1 - 1)
        df[i+1] = -200 * (x1**2 - x2) 

    return 1.0/const * df

def alt_rosenbrock_eval(H, persis_info, sim_specs, _):

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
