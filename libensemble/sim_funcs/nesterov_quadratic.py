import numpy as np


def EvaluateFunction(x, component):
    """
    Evaluates the chained Rosenbrock function
    """
    n = len(x)

    i = component
    assert 0 <= i <= n

    if i == 0:
        x_1 = x[i]
        f_i = 0.5 * x_1**2 - x_1
    elif i == n:
        x_n = x[-1]
        f_i = 0.5 * x_n**2
    else:
        x_1 = x[i - 1]
        x_2 = x[i]
        f_i = 0.5 * (x_2 - x_1) ** 2

    return f_i


def EvaluateJacobian(x, component=np.nan):
    """
    Evaluates the chained Rosenbrock Jacobian
    """
    n = len(x)
    df = np.zeros(n, dtype=float)

    i = component
    assert 0 <= i <= n

    if i == 0:
        x_1 = x[i]
        df[0] = x_1 - 1
    elif i == n:
        x_n = x[-1]
        df[-1] = x_n
    else:
        x_1 = x[i - 1]
        x_2 = x[i]

        df[i - 1] = x_1 - x_2
        df[i] = x_2 - x_1

    return df


def nesterov_quadratic_eval(H, persis_info, sim_specs, _):
    batch = len(H["x"])
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        obj_component = H["obj_component"][i]  # which f_i

        if H[i]["get_grad"]:
            H_o["gradf_i"][i] = EvaluateJacobian(x, obj_component)
        else:
            H_o["f_i"][i] = EvaluateFunction(x, obj_component)

    return H_o, persis_info
