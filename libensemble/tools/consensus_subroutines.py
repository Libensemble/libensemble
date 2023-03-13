"""
This file contains many common subroutines used in distributed optimization
libraries, including collecting all the sum of {f_i}'s, collecting the
gradients, and conducting the consensus step (i.e., take linear combination of
your neighbors' $x$ values.
"""

import nlopt
import numpy as np
import scipy.sparse as spp

from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport


def print_final_score(x, f_i_idxs, gen_specs, libE_info):
    """This function is called by a gen so that the alloc will collect
        all the {f_i}'s and print their sum.

    Parameters
    ----------
    - x : np.ndarray
        Input solution vector
    - f_i_idxs : np.ndarray
        Which {f_i}'s this calling gen is responsible for
    - gen_specs, libE_info :
        Used to communicate
    """
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # evaluate { f_i(x) } first
    H_o = np.zeros(len(f_i_idxs), dtype=gen_specs["out"])
    H_o["x"][:] = x
    H_o["consensus_pt"][:] = False
    H_o["obj_component"][:] = f_i_idxs
    H_o["get_grad"][:] = False
    H_o = np.reshape(H_o, newshape=(-1,))

    tag, Work, calc_in = ps.send_recv(H_o)

    if tag in [PERSIS_STOP, STOP_TAG]:
        return

    f_is = calc_in["f_i"]
    F_i = np.sum(f_is)

    # get alloc to print sum of f_i
    H_o = np.zeros(1, dtype=gen_specs["out"])
    H_o["x"][0] = x
    H_o["f_i"][0] = F_i
    H_o["eval_pt"][0] = True
    H_o["consensus_pt"][0] = True

    ps.send_recv(H_o)


def get_func_or_grad(x, f_i_idxs, gen_specs, libE_info, get_grad):
    """This function is called by a gen to retrieve the function or gradient
        of the sum of {f_i}'s via the sim.

    Parameters
    ----------
    - x : np.ndarray
        Input solution vector
    - f_i_idxs : np.ndarray
        Which {f_i}'s this calling gen is responsible for
    - gen_specs, libE_info :
        Used to communicate
    - get_grad : bool
        True if we want gradient, otherwise returns function eval
    """
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    H_o = np.zeros(len(f_i_idxs), dtype=gen_specs["out"])
    H_o["x"][:] = x
    H_o["consensus_pt"][:] = False
    H_o["obj_component"][:] = f_i_idxs
    H_o["get_grad"][:] = get_grad
    H_o = np.reshape(H_o, newshape=(-1,))  # unfold into 1d array

    tag, Work, calc_in = ps.send_recv(H_o)

    if tag in [STOP_TAG, PERSIS_STOP]:
        return tag, None

    if get_grad:
        gradf_is = calc_in["gradf_i"]
        gradf = np.sum(gradf_is, axis=0)
        return tag, gradf
    else:
        f_is = calc_in["f_i"]
        f = np.sum(f_is)
        return tag, f


def get_func(x, f_i_idxs, gen_specs, libE_info):
    return get_func_or_grad(x, f_i_idxs, gen_specs, libE_info, get_grad=False)


def get_grad(x, f_i_idxs, gen_specs, libE_info):
    return get_func_or_grad(x, f_i_idxs, gen_specs, libE_info, get_grad=True)


def get_grad_locally(x, f_i_idxs, df):
    """This function is called by a gen to locally compute gradients of
        the sum of {f_i}'s. Unlike `get_grad`, this function does not
        use the sim, but instead evaluates the gradient using the input @df.

    Parameters
    ----------
    - x : np.ndarray
        Input solution vector
    - f_i_idxs : np.ndarray
        Which {f_i}'s this calling gen is responsible for
    - df : func
        Function that returns gradient. Must take in as parameters input @x and
        index @i (i.e., which f_i to take gradient of)
    """
    gradf = np.zeros(len(x), dtype=float)
    for i in f_i_idxs:
        gradf += df(x, i)

    return gradf


def get_neighbor_vals(x, local_gen_id, A_gen_ids_no_local, gen_specs, libE_info):
    """Sends local gen data (@x) and retrieves neighbors local data.
        Sorts the data so the gen ids are in increasing order

    Parameters
    ----------
    x : np.ndarray
        - local input variable

    local_gen_id : int
        - this gen's gen_id

    A_gen_ids_local : int
        - expected neighbor's gen ids, not including local gen id

    gen_specs, libE_info : ?
        - objects to communicate and construct mini History array

    Returns
    -------
    X : np.ndarray
        - 2D array of neighbors and local x values sorted by gen_ids
    """
    H_o = np.zeros(1, dtype=gen_specs["out"])
    H_o["x"][0] = x
    H_o["consensus_pt"][0] = True
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    tag, Work, calc_in = ps.send_recv(H_o)
    if tag in [STOP_TAG, PERSIS_STOP]:
        return tag, None

    neighbor_X = calc_in["x"]
    neighbor_gen_ids = calc_in["gen_worker"]

    assert local_gen_id not in neighbor_gen_ids, "Local data should not be sent back from manager"

    assert np.array_equal(A_gen_ids_no_local, neighbor_gen_ids), "Expected gen_ids {}, received {}".format(
        A_gen_ids_no_local, neighbor_gen_ids
    )

    X = np.vstack((neighbor_X, x))
    gen_ids = np.append(neighbor_gen_ids, local_gen_id)

    # sort data (including local) in corresponding gen_id increasing order
    X[:] = X[np.argsort(gen_ids)]

    return tag, X


def get_consensus_gradient(x, gen_specs, libE_info):
    """Sends local gen data (@x) and retrieves neighbors local data,
        and takes sum of the neighbors' x's, which is equivalent to taking
        the gradient of consensus term for this particular node/agent.

        This function is equivalent to the @get_neighbor_vals function, but is
        less general, i.e., when we need only take a sum rather than a linear
        combination of our neighbors.

    Parameters
    ----------
    x : np.ndarray
        - local input variable

    Returns
    -------
     : np.ndarray
        - Returns this node's corresponding gradient of consensus
    """
    tag = None
    H_o = np.zeros(1, dtype=gen_specs["out"])
    H_o["x"][0] = x
    H_o["consensus_pt"][0] = True
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    tag, Work, calc_in = ps.send_recv(H_o)

    if tag in [PERSIS_STOP, STOP_TAG]:
        return tag, np.zeros(len(x))

    neighbor_X = calc_in["x"]
    num_neighbors = len(neighbor_X)
    grad_cons = (num_neighbors * x) - np.sum(neighbor_X, axis=0)

    return tag, grad_cons


def get_k_reach_chain_matrix(n, k):
    """Constructs adjacency matrix for a chain matrix where the ith vertex can
    reach vertices that are at most @k distances from them (does not wrap around),
    where the distance is based on the absolute difference between vertices'
    indexes.
    """
    assert 1 <= k <= n - 1

    half_of_diagonals = [np.ones(n - k + j) for j in range(k)]
    half_of_indices = np.arange(1, k + 1)
    all_of_diagonals = half_of_diagonals + half_of_diagonals[::-1]
    all_of_indices = np.append(-half_of_indices[::-1], half_of_indices)
    A = spp.csr_matrix(spp.diags(all_of_diagonals, all_of_indices))
    return A


def get_doubly_stochastic(A):
    """Generates a doubly stochastic matrix where
    (i) S_ii > 0 for all i
    (ii) S_ij > 0 if and only if (i, j) in E

    Parameters
    ----------
    A : np.ndarray
        - adjacency matrix

    Returns
    -------
    x : scipy.sparse.csr_matrix
    """
    np.random.seed(0)
    n = A.shape[0]
    x = np.multiply(A.toarray() != 0, np.random.random((n, n)))
    x = x + np.diag(np.random.random(n) + 1e-4)

    rsum = np.zeros(n)
    csum = np.zeros(n)
    tol = 1e-15

    while (np.any(np.abs(rsum - 1) > tol)) | (np.any(np.abs(csum - 1) > tol)):
        x = x / x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)

    X = spp.csr_matrix(x)
    return X


"""
The remaining functions below don't help with consensus, but help with the
regression tests. One can move these functions to a different Python file
if need be.
"""


def readin_csv(fname):
    """Parses breast-cancer dataset
        (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)
        for SVM.

    Parameters
    ----------
    - fname : str
        file name containing data

    Returns
    -------
    - labels : np.ndarray, (m,)
        1D with the label of each vector
    - datas : np.ndarray (2D), (m, n)
        2D array (matrix) with the collection of dataset
    """
    n = 569
    try:
        fp = open(fname, "r+")
    except FileNotFoundError:
        print(
            "# Missing file 'wdbc.data' (must be placed in same directory where you run 'mpirun -np ...'). "
            "File can be downloaded from https://tinyurl.com/3a2ttj5a. "
            "Creating artificial dataset instead."
        )

        m = 30
        np.random.seed(0)
        label = 2 * np.random.randint(low=0, high=1, size=m) - 1
        datas = (5 * np.random.random(size=(n, m))).astype("int")

        return label, datas

    label = np.zeros(n, dtype="int")
    datas = np.zeros((n, 30))
    i = 0

    for line in fp.readlines():
        line = line.rsplit()[0]
        data = line.split(",")
        label[i] = data[1] == "M"
        datas[i, :] = [float(val) for val in data[2:32]]
        i += 1
    fp.close()

    assert i == n, f"Expected {n} datapoints, recorded {i}"

    return label, datas


def gm_opt(b, m):
    """Computes optimal geometric median score

    Parameters
    ----------
    - b : np.ndarray, (m*n,)
        1D array concatenating @m vectors of size @n, i.e., [x_1, x_2,..., x_m]
    - m : int
        number of vectors
    """

    n = len(b) // m
    assert len(b) == m * n

    ones_m = np.ones((m, 1))

    def obj_fn(x, B, m):
        X = np.outer(ones_m, x.T)
        return (1 / m) * np.sum(np.linalg.norm(X - B, 2, axis=1))

    B = np.reshape(b, newshape=(m, n))

    opt = nlopt.opt(nlopt.LN_COBYLA, n)
    opt.set_min_objective(lambda beta, grad: obj_fn(beta, B, m))
    opt.set_xtol_rel(1e-8)

    opt.optimize(np.zeros(n))
    minf = opt.last_optimum_value()

    return minf


def regls_opt(X, y, c, reg=None):
    """Computes optimal linear regression with l2 regularization

    Parameters
    ----------
    - X, y : np.ndarray
        2D matrix, 1D matrix, where we want to solve optimally for theta so that
        $y \\\\approx X.dot(theta)$
    - c : float
        Scalar term for regularization
    - reg : str
        Denotes which regularization to use. Either 'l1', 'l2', or None
    """
    if reg == "l1":
        p = 1
    elif reg == "l2":
        p = 2
    elif reg is None:
        p = -1
    else:
        assert False, f'illegal regularization "{reg}"'

    def obj_fn(X, y, beta, c, p):
        m = X.shape[0]
        if p == 1:
            return (1 / m) * np.linalg.norm(X @ beta - y, ord=2) ** 2 + c * np.linalg.norm(beta, ord=1)
        return (1 / m) * np.linalg.norm(X @ beta - y, ord=2) ** 2 + c * np.linalg.norm(beta, ord=2) ** 2

    # already X.T
    d, m = X.shape

    opt = nlopt.opt(nlopt.LN_COBYLA, d)
    opt.set_min_objective(lambda beta, grad: obj_fn(X.T, y, beta, c, p))
    opt.set_xtol_rel(1e-8)

    opt.optimize(np.zeros(d))
    minf = opt.last_optimum_value()

    return minf


def log_opt(X, y, c, reg=None):
    """Computes optimal linear regression with l2 regularization. See, for
        reference,
        https://www.cvxpy.org/examples/machine_learning/logistic_regression.html

    Parameters
    ----------
    - X, y : np.ndarray
        2D matrix, 1D matrix, defining the logisitic regression problem
    - c : float
        Scalar term for regularization
    - reg : str
        Denotes which regularization to use. Either 'l1', 'l2', or None
    """
    assert reg == "l2", "Only l2 regularization allowed"
    # if reg == 'l1':
    #     p = 1
    # elif reg == 'l2':
    p = 2
    # elif reg is None:
    #     p = 0
    # else:
    #     assert False, 'illegal regularization mode, "{}"'.format(reg)

    def obj_fn(X, y, beta, c, p):
        m = X.shape[0]
        # if p == 0:
        #     reg = 0
        # if p == 1:
        #     reg = c * np.linalg.norm(beta, 1)
        # elif p == 2:
        reg = c * np.linalg.norm(beta, 2) ** 2
        # Note that, cp.logistic(x) == log(1+e^x)
        return (1 / m) * np.sum(np.log(1 + np.exp(np.multiply(-y, X @ beta)))) + reg

    d, m = X.shape

    opt = nlopt.opt(nlopt.LN_COBYLA, d)
    opt.set_min_objective(lambda beta, grad: obj_fn(X.T, y, beta, c, p))
    opt.set_xtol_rel(1e-8)

    opt.optimize(np.zeros(d))
    minf = opt.last_optimum_value()

    return minf


def svm_opt(X, b, c, reg="l1"):
    """Computes optimal support vector machine (SVM) with l1 regularization.

    Parameters
    ----------
    - X, b : np.ndarray
        2D matrix, 1D matrix, defining the SVM problem
    - c : float
        Scalar term for regularization
    - reg : str
        Denotes which regularization to use. Either 'l1', 'l2', or None
    """
    assert reg == "l1", "Only l1 regularization allowed"

    # if reg == 'l1':
    p = 1
    # elif reg == 'l2':
    #     p = 2
    # elif reg is None:
    #     p = 0
    # else:
    #     assert False, 'illegal regularization mode, "{}"'.format(reg)

    def obj_fn(X, b, theta, c, p):
        # if p == 0:
        #     reg = 0
        # if p == 1:
        reg = c * np.linalg.norm(theta, 1)
        # if p == 2:
        #     reg = c * np.linalg.norm(theta, 2)**2
        return np.sum(np.maximum(0, 1 - np.multiply(b, X @ theta))) + reg

    d, m = X.shape
    opt = nlopt.opt(nlopt.LN_COBYLA, d)
    opt.set_min_objective(lambda theta, grad: obj_fn(X.T, b, theta, c, p))
    opt.set_xtol_rel(1e-8)

    opt.optimize(np.zeros(d))
    minf = opt.last_optimum_value()

    return minf
