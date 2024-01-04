"""Persistent generator exposing gpCAM functionality"""

import copy
import time

import numpy as np
from gpcam import GPOptimizer as GP
from scipy.spatial.distance import pdist, squareform

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = [
    "persistent_gpCAM_simple",
    "persistent_gpCAM_ask_tell",
]


def _initialize_gpcAM(user_specs, libE_info):
    """Extract user params"""
    b = user_specs["batch_size"]
    lb = np.array(user_specs["lb"])
    ub = np.array(user_specs["ub"])
    n = len(lb)  # dimension
    assert isinstance(b, int), "Batch size must be an integer"
    assert isinstance(n, int), "Dimension must be an integer"
    assert isinstance(lb, np.ndarray), "lb must be a numpy array"
    assert isinstance(ub, np.ndarray), "ub must be a numpy array"

    all_x = np.empty((0, n))
    all_y = np.empty((0, 1))

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    np.random.seed(0)

    return b, n, lb, ub, all_x, all_y, ps


def _generate_nd_mesh(lb, ub, num_points=10):
    """
    Generate a mesh of points in n-dimensional space over a hypercube defined by lb and ub.

    :param lb: Lower bound (n-dimensional array).
    :param ub: Upper bound (n-dimensional array).
    :param num_points: Number of points to generate in each dimension.
    :return: A mesh of points as a numpy array.
    """
    # Generate grids for each dimension
    grids = [np.linspace(lb[i], ub[i], num_points) for i in range(len(lb))]

    # Create a meshgrid
    mesh = np.meshgrid(*grids)

    # Convert the meshgrid to a list of points
    points = np.stack(mesh, axis=-1).reshape(-1, len(lb))
    D = squareform(pdist(points))
    return points, D


def _update_gp_and_eval_var(all_x, all_y, x_for_var):
    """
    Update the GP using the points in all_x and their function values in
    all_y. (We are assuming deterministic values in all_y, so we set the noise
    to be 1e-8 when build the GP.) Then evaluates the posterior covariance at
    points in x_for_var.
    """
    my_gp2S = GP(all_x, all_y, noise_variances=1e-8 * np.ones(len(all_y)))
    my_gp2S.train(max_iter=2)
    var_rand = my_gp2S.posterior_covariance(x_for_var, variance_only=True)["v(x)"]
    # print(np.max(var_rand))

    return var_rand


def find_eligible_points(X, D, F, r):
    """
    Find points in X such that no point has another point within distance r with a larger F value.

    :param X: A 2D numpy array where each row represents a point.
    :param D: Pairwise distance matrix for points in X.
    :param F: Function values for each point in X.
    :param r: Radius constraint.
    :return: Indices of the eligible points in the original X.
    """
    # Sort points by their function values in descending order
    sorted_indices = np.argsort(-F)

    sorted_X = copy.deepcopy(X)
    sorted_D = copy.deepcopy(D)

    sorted_X = X[sorted_indices]
    sorted_D = D[:, sorted_indices][sorted_indices]

    eligible_indices = []
    for idx in range(len(sorted_X)):
        # Check if this point is within r distance of any point already added
        if not any(sorted_D[idx, :idx] < r):
            eligible_indices.append(sorted_indices[idx])

    return eligible_indices


def persistent_gpCAM_simple(H_in, persis_info, gen_specs, libE_info):
    """
    This generation function constructs a global surrogate of `f` values.
    It is a batched method that produces a first batch uniformly random from
    (lb, ub) and on following iterations samples the GP posterior covariance
    function to find sample points.

    .. seealso::
        `test_gpCAM.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_gpCAM.py>`_
    """  # noqa
    U = gen_specs["user"]

    batch_size, n, lb, ub, all_x, all_y, ps = _initialize_gpcAM(U, libE_info)

    # Send batches until manager sends stop tag
    tag = None
    persis_info["max_variance"] = []

    if U.get("use_grid"):
        x_for_var, D = _generate_nd_mesh(lb, ub)
        vals_above_diagonal = D[np.triu_indices(len(x_for_var), 1)]
        r_high_init = np.max(vals_above_diagonal)
        r_low_init = np.min(vals_above_diagonal)

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if all_x.shape[0] == 0:
            x_new = persis_info["rand_stream"].uniform(lb, ub, (batch_size, n))
        else:
            if not U.get("use_grid"):
                x_for_var = persis_info["rand_stream"].uniform(lb, ub, (10 * batch_size, n))
            var_rand = _update_gp_and_eval_var(all_x, all_y, x_for_var)
            persis_info["max_variance"].append(np.max(var_rand))

            if U.get("use_grid"):
                r_high = r_high_init
                r_low = r_low_init
                new_inds = []
                r_cand = r_high  # Let's start with a large radius and stop when we have batchsize points

                while len(new_inds) < batch_size:
                    new_inds = find_eligible_points(x_for_var, D, var_rand, r_cand)
                    if len(new_inds) < batch_size:
                        r_high = r_cand
                    r_cand = (r_high + r_low) / 2.0

                x_new = x_for_var[new_inds[:batch_size]]
            else:
                x_new = x_for_var[np.argsort(var_rand)[-batch_size:]]

        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        H_o["x"] = x_new
        tag, Work, calc_in = ps.send_recv(H_o)

        if calc_in is not None:
            all_x = np.vstack((all_x, x_new))
            all_y = np.vstack((all_y, np.atleast_2d(calc_in["f"]).T))

    # If final points are sent with PERSIS_STOP, update model and get final var_rand
    if calc_in is not None:
        # H_o not updated by default - is persis_info
        x_for_var = persis_info["rand_stream"].uniform(lb, ub, (10 * batch_size, n))
        var_rand = _update_gp_and_eval_var(all_x, all_y, x_for_var)
        persis_info["max_variance"].append(np.max(var_rand))

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_gpCAM_ask_tell(H_in, persis_info, gen_specs, libE_info):
    """
    Like persistent_gpCAM_simple, this generation function constructs a global
    surrogate of `f` values. It also aa batched method that produces a first batch
    uniformly random from (lb, ub). On subequent iterations, it calls an
    optimization method to produce the next batch of points. This optimization
    might be too slow (relative to the simulation evaluation time) for some use cases.

    .. seealso::
        `test_gpCAM.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_gpCAM.py>`_
    """  # noqa

    batch_size, n, lb, ub, all_x, all_y, ps = _initialize_gpcAM(gen_specs["user"], libE_info)

    H_o = np.zeros(batch_size, dtype=gen_specs["out"])
    x_new = persis_info["rand_stream"].uniform(lb, ub, (batch_size, n))
    H_o["x"] = x_new

    tag, Work, calc_in = ps.send_recv(H_o)

    first_call = True
    while tag not in [STOP_TAG, PERSIS_STOP]:
        all_x = np.vstack((all_x, x_new))
        all_y = np.vstack((all_y, np.atleast_2d(calc_in["f"]).T))

        if first_call:
            # Initialize GP
            my_gp2S = GP(all_x, all_y, noise_variances=1e-8 * np.ones(len(all_y)))
            first_call = False
        else:
            my_gp2S.tell(all_x, all_y, noise_variances=1e-8 * np.ones(len(all_y)))

        my_gp2S.train()

        start = time.time()
        x_new = my_gp2S.ask(
            bounds=np.column_stack((lb, ub)),
            n=batch_size,
            pop_size=batch_size,
            max_iter=1,
        )["x"]
        print(f"Ask time:{time.time() - start}")
        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        H_o["x"] = x_new

        tag, Work, calc_in = ps.send_recv(H_o)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
