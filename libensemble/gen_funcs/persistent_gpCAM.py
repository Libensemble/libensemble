"""Persistent generator exposing gpCAM functionality"""

import time
import numpy as np
from gpcam import GPOptimizer as GP

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


def _generate_mesh(lb, ub, num_points=10):
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

    return points


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


def calculate_grid_distances(lb, ub, num_points):
    """Calculate mininum and maximum distances between points in grid"""
    num_points = [num_points] * len(lb)
    spacings = [(ub[i] - lb[i]) / (num_points[i] - 1) for i in range(len(lb))]
    min_distance = min(spacings)
    max_distance = np.sqrt(sum([(ub[i] - lb[i]) ** 2 for i in range(len(lb))]))
    return min_distance, max_distance


def is_point_far_enough(point, eligible_points, r):
    """Check if point is at least r distance away from all points in eligible_points."""
    for ep in eligible_points:
        if np.linalg.norm(point - ep) < r:
            return False
    return True


def _find_eligible_points(x_for_var, sorted_indices, r, batch_size):
    """
    Find points in X such that no point has another point within distance r with a larger F value.

    :param x_for_var: positions of each point mesh
    :param sorted_indices: Indices sorted based on variance (highest to lowest).
    :param r: Radius constraint.
    :param batch_size: Number of points requested
    :return: The eligible points in the original X.
    """
    eligible_points = []
    for idx in sorted_indices:
        point = x_for_var[idx]
        if is_point_far_enough(point, eligible_points, r):
            eligible_points.append(point)
            if len(eligible_points) == batch_size:
                break
    return np.array(eligible_points)


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
        num_points = 10
        x_for_var = _generate_mesh(lb, ub)
        r_low_init, r_high_init = calculate_grid_distances(lb, ub, num_points)

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
                x_new = []
                r_cand = r_high  # Let's start with a large radius and stop when we have batchsize points

                sorted_indices = np.argsort(-var_rand)
                while len(x_new) < batch_size:
                    x_new = _find_eligible_points(x_for_var, sorted_indices, r_cand, batch_size)
                    if len(x_new) < batch_size:
                        r_high = r_cand
                    r_cand = (r_high + r_low) / 2.0
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
        if not U.get("use_grid"):
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
