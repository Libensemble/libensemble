import numpy as np

const = 1500


def EvaluateFunction(x, component=np.nan):
    """
    Evaluates the chained Rosenbrock function
    """

    assert not np.isnan(component), "Must give a component"

    i = component
    x1 = x[i]
    x2 = x[i + 1]
    f = 100 * (x1**2 - x2) ** 2 + (x1 - 1) ** 2

    return f


def EvaluateJacobian(x, component=np.nan):
    """
    Evaluates the chained Rosenbrock Jacobian
    """

    df = np.zeros(len(x), dtype=float)

    assert not np.isnan(component), "Must give a component"

    i = component
    x1 = x[i]
    x2 = x[i + 1]

    df[i] = 400 * x1 * (x1**2 - x2) + 2 * (x1 - 1)
    df[i + 1] = -200 * (x1**2 - x2)

    return 1.0 / const * df


def alt_rosenbrock_eval(H, persis_info, sim_specs, _):
    batch = len(H["x"])
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        obj_component = H["obj_component"][i]  # which f_i
        if H[i]["get_grad"]:
            H_o["gradf_i"][i] = EvaluateJacobian(x, obj_component)
        else:
            H_o["f_i"][i] = EvaluateFunction(x, obj_component)

    return H_o, persis_info
