import numpy as np


def EvaluateFunction(theta, component, X, y, c, reg):
    """
    Evaluates linear regression with l2 regularization
    """
    i = component
    m = len(y)

    y_i = y[i]
    X_i = X[:, i]

    base = np.exp(-y_i * np.dot(X_i, theta))

    f_i = (1 / m) * np.log(1 + base)

    assert reg == "l2", "Only l2 regularization allowed"
    # if reg is None:
    #     reg_val = 0
    # elif reg == 'l1':
    #     reg_val = (c/m) * np.sum(np.abs(theta))
    reg_val = (c / m) * np.dot(theta, theta)

    return f_i + reg_val


def EvaluateJacobian(theta, component, X, y, c, reg):
    """
    Evaluates linear regression with l2 regularization
    """

    i = component
    m = len(y)

    y_i = y[i]
    X_i = X[:, i]

    base = np.exp(-y_i * np.dot(X_i, theta))

    df_i = (1 / m) * (-y_i * base) / (1 + base) * X_i

    assert reg == "l2", "Only l2 regularization allowed"
    # if reg is None:
    #     reg_val = 0
    # elif reg == 'l1':
    #     reg_val = (c/m) * np.sign(theta)
    reg_val = (2 * c / m) * theta

    return df_i + reg_val


def logistic_regression_eval(H, persis_info, sim_specs, _):
    X = persis_info["params"]["X"]
    y = persis_info["params"]["y"]
    c = persis_info["params"]["c"]
    reg = persis_info["params"].get("reg", None)

    assert (reg is None) or (reg == "l1") or (reg == "l2"), f"Incompatible regularization {reg}"

    batch = len(H["x"])
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        obj_component = H["obj_component"][i]  # which f_i

        if H[i]["get_grad"]:
            H_o["gradf_i"][i] = EvaluateJacobian(x, obj_component, X, y, c, reg)
        else:
            H_o["f_i"][i] = EvaluateFunction(x, obj_component, X, y, c, reg)

    return H_o, persis_info
