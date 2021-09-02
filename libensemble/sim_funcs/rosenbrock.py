import numpy as np


def EvaluateFunction(x, component):
    """
    Evaluates the chained Rosenbrock function
    """

    assert len(x) % 2 == 0, print("must be even lengthed input vector")

    n = len(x) // 2

    if np.isnan(component):
        f1 = 100 * np.power(np.power(x[::2], 2) - x[1::2], 2)
        f2 = np.power(x[::2]-np.ones(n), 2)
        f = f1 + f2
    else:
        i = component
        x1 = x[2*i]
        x2 = x[2*i+1]
        f = 100 * (x1**2 - x2)**2 + (x1-1)**2

    return f


def EvaluateJacobian(x, component, const):
    """
    Evaluates the chained Rosenbrock Jacobian
    """

    assert len(x) % 2 == 0, print("must be even lengthed input vector")

    n = len(x) // 2
    df = np.zeros(len(x), dtype=float)

    if np.isnan(component):
        df[::2] = 400 * np.multiply(x[::2], np.power(x[::2], 2) - x[1::2]) + 2 * (x[::2] - np.ones(n))
        df[1::2] = -200 * (np.power(x[::2], 2) - x[1::2])

    else:
        i = component
        x1 = x[2*i]
        x2 = x[2*i+1]

        df[2*i] = 400 * x1 * (x1**2 - x2) + 2 * (x1 - 1)
        df[2*i+1] = -200 * (x1**2 - x2)

    return 1.0/const * df


def rosenbrock_eval(H, persis_info, sim_specs, _):

    if 'params' in persis_info:
        const = persis_info['params'].get('const', 1000)
    else:
        const = 1000

    batch = len(H['x'])
    H_o = np.zeros(batch, dtype=sim_specs['out'])

    for i, x in enumerate(H['x']):
        if 'obj_component' in H.dtype.fields:
            obj_component = H['obj_component'][i]  # which f_i

            if H[i]['get_grad']:
                H_o['gradf_i'][i] = EvaluateJacobian(x, obj_component, const)
            else:
                H_o['f_i'][i] = EvaluateFunction(x, obj_component)

        else:

            if 'grad' in H.dtype.fields:
                H_o['grad'][i] = EvaluateJacobian(x, np.nan, const)

            H_o['f'][i] = EvaluateFunction(x, np.nan)
    return H_o, persis_info
