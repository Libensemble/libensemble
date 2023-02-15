__all__ = ["geomedian_eval"]
import numpy as np
import numpy.linalg as la


def EvaluateFunction(x, component, B):
    """
    Evaluates the sum of squares-variant chwirut function
    """
    m = B.shape[0]
    assert B.shape[1] == len(x)
    assert 0 <= component <= m - 1
    i = component
    b_i = B[i]

    f_i = 1.0 / m * la.norm(x - b_i)
    return f_i


def EvaluateJacobian(x, component, B):
    """
    Evaluates the sum of squares-variant chwirut Jacobian
    """
    m = B.shape[0]
    assert B.shape[1] == len(x)
    assert 0 <= component <= m - 1
    i = component
    b_i = B[i]

    df_i = 1.0 / m * (x - b_i) / la.norm(x - b_i)
    return df_i


def geomedian_eval(H, persis_info, sim_specs, _):
    B = persis_info["params"]["B"]

    num_xs = len(H["x"])  # b==1 always?
    H_o = np.zeros(num_xs, dtype=sim_specs["out"])

    for k, x in enumerate(H["x"]):
        i = H[k]["obj_component"]  # f_i

        if H[k]["get_grad"]:
            H_o["gradf_i"][k] = EvaluateJacobian(x, i, B)
        else:
            H_o["f_i"][k] = EvaluateFunction(x, i, B)

    return H_o, persis_info
