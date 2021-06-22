__all__ = ['geomedian_eval']
import numpy as np
import numpy.linalg as la

m = 2

def EvaluateFunction(x, component):
    """
    Evaluates the sum of squares-variant chwirut function
    """
    assert 0 <= component <= m-1
    i = component
    np.random.seed(i)
    b_i = np.random.random(len(x))
    # b_i = np.ones(len(x))

    f_i = 1.0/m * la.norm(x-b_i)
    return f_i

def EvaluateJacobian(x, component):
    """
    Evaluates the sum of squares-variant chwirut Jacobian
    """
    assert 0 <= component <= m-1
    i = component

    np.random.seed(i)
    b_i = np.random.random(len(x))
    # b_i = np.ones(len(x))

    df_i = 1.0/m * (x-b_i)/la.norm(x-b_i)
    return df_i

def geomedian_eval(H, persis_info, sim_specs, _):

    b = len(H['x']) # b==1 always?
    O = np.zeros(b, dtype=sim_specs['out'])

    for k, x in enumerate(H['x']):
        i = H[k]['obj_component']   # f_i

        if H[k]['get_grad']:
            O['gradf_i'][k] = EvaluateJacobian(x, i)
        else:
            O['f_i'][k] = EvaluateFunction(x, i)

    return O, persis_info
