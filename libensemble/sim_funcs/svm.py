import numpy as np


def EvaluateFunction(theta, component, X, b, c, reg):
    """
    Evaluates svm with l1 regularization
    """
    i = component
    m = len(b)

    b_i = b[i]
    X_i = X[:, i]

    f_i = max(0, 1 - b_i * np.dot(X_i, theta))

    assert reg == "l1", "Only l1 regularization allowed"

    # if reg is None:
    #     reg_val = 0
    # elif reg == 'l1':
    reg_val = (c / m) * np.sum(np.abs(theta))
    # else:
    #     reg_val = (c/m) * np.dot(theta, theta)

    return f_i + reg_val


def EvaluateJacobian(theta, component, X, b, c, reg):
    """
    Evaluates svm with l1 regularization
    """

    i = component
    m = len(b)

    b_i = b[i]
    X_i = X[:, i]

    score = b_i * np.dot(X_i, theta)

    if score >= 1:
        df_i = np.zeros(len(theta))
    else:
        df_i = -b_i * X_i

    assert reg == "l1", "Only l1 regularization allowed"

    # if reg is None:
    #     reg_val = 0
    # elif reg == 'l1':
    reg_val = (c / m) * np.sign(theta)
    # else:
    #     reg_val = (2*c/m) * theta

    return df_i + reg_val


def svm_eval(H, persis_info, sim_specs, _):
    X = persis_info["params"]["X"]
    b = persis_info["params"]["b"]
    c = persis_info["params"]["c"]
    reg = persis_info["params"].get("reg", None)

    assert (reg is None) or (reg == "l1") or (reg == "l2"), f"Incompatible regularization {reg}"

    batch = len(H["x"])
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i, x in enumerate(H["x"]):
        obj_component = H["obj_component"][i]  # which f_i

        if H[i]["get_grad"]:
            H_o["gradf_i"][i] = EvaluateJacobian(x, obj_component, X, b, c, reg)
        else:
            H_o["f_i"][i] = EvaluateFunction(x, obj_component, X, b, c, reg)

    return H_o, persis_info
