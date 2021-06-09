import numpy as np

def EvaluateFunction(x, component=np.nan):
    """
    Evaluates the chained Rosenbrock function
    """

    assert len(x) % 2 == 0, print("must be even lengthed input vector")

    n = len(x) // 2

    if np.isnan(component):
        f1 = 100 * np.power( np.power(x[::2],2) - x[1::2], 2)
        f2 = np.power(x[::2]-np.ones(n), 2)
        f =  f1 + f2
    else:
        i = component
        x1 = x[2*i]
        x2 = x[2*i+1]
        f = 100 * (x1**2 - x2)**2 + (x1-1)**2

    return f


def EvaluateJacobian(x, component=np.nan):
    """
    Evaluates the chained Rosenbrock Jacobian
    """

    assert len(x) % 2 == 0, print("must be even lengthed input vector")

    n = len(x) // 2

    if np.isnan(component):
        df = np.zeros(len(x), dtype=float)
        df[::2] = 400 * np.multiply(x[::2], np.power(x[::2], 2) - x[1::2]) \
                + 2 * ( x[::2] - np.ones(n) )

        df[1::2] = -200 * (np.power(x[::2], 2) - x[1::2])
    else:
        i = component
        x1 = x[2*(i//2)]
        x2 = x[2*(i//2)+1]

        if i % 2 == 0:
            df = 400 * x1 * (x1**2 - x2) + 2 * (x1 - 1)
        else:
            df = -200 * (x1**2 - x2)

    return df

def rosenbrock_eval(H, persis_info, sim_specs, _):

    batch = len(H['x'])
    O = np.zeros(batch, dtype=sim_specs['out'])

    for i, x in enumerate(H['x']):
        obj_component_idx = H['obj_component'][i]

        if 'f_i' in O.dtype.names:
            # TODO: bandaid fix, find how to not eval f_i when we only want gradf_i
            if obj_component_idx < len(x)//2:
                O['f_i'][i] = EvaluateFunction(x, obj_component_idx)

        if 'gradf_i' in O.dtype.names:
            O['gradf_i'][i] = EvaluateJacobian(x, obj_component_idx)

    return O, persis_info
