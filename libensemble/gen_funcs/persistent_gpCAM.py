"""Persistent generator exposing gpCAM functionality"""

import time

import numpy as np
from gpcam import GPOptimizer as GP
from numpy.lib.recfunctions import repack_fields

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = [
    "persistent_gpCAM",
    "persistent_gpCAM_covar",
]


def _initialize_gpcAM(user_specs, libE_info, persis_info):
    """Extract user params"""
    rng_seed = user_specs.get("rng_seed")  # Will default to None
    rng = persis_info.get("rand_stream") or np.random.default_rng(rng_seed)
    b = user_specs["batch_size"]
    lb = np.array(user_specs["lb"])
    ub = np.array(user_specs["ub"])
    n = len(lb)  # no. of dimensions
    init_x = np.empty((0, n))
    init_y = np.empty((0, 1))
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)  # init comms
    return rng, b, n, lb, ub, init_x, init_y, ps


def _read_testpoints(U):
    """Read numpy file containing evaluated points for measuring GP error"""
    test_points_file = U.get("test_points_file")
    if test_points_file is None:
        return None
    test_points = np.load(test_points_file)

    # Remove any NaNs
    nan_indices = [i for i, fval in enumerate(test_points["f"]) if np.isnan(fval)]
    test_points = np.delete(test_points, nan_indices, axis=0)

    # In case large fields we don't need
    test_points = repack_fields(test_points[["x", "f"]])

    return test_points


def _compare_testpoints(my_gp, test_points, persis_info):
    """Compare model at test points"""
    if test_points is None:
        return
    f_est = my_gp.posterior_mean(test_points["x"])["f(x)"]
    mse = np.mean((f_est - test_points["f"]) ** 2)
    persis_info.setdefault("mean_squared_error", []).append(float(mse))


def _update_gp(my_gp, x_new, y_new, test_points, persis_info, noise):
    """Update Gaussian process with new points and train"""
    noise_arr = noise * np.ones(len(y_new))  # Initializes noise
    if my_gp is None:
        my_gp = GP(x_new, y_new.flatten(), noise_variances=noise_arr)
    else:
        my_gp.tell(x_new, y_new.flatten(), noise_variances=noise_arr, append=True)
    my_gp.train()

    if test_points is not None:
        _compare_testpoints(my_gp, test_points, persis_info)

    return my_gp


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


def _eval_var(my_gp, all_x, all_y, x_for_var, test_points, persis_info):
    """
    Evaluate the posterior covariance at points in x_for_var.
    If we have test points, calculate mean square error at those points.
    """
    # Obtain covariance in groups to prevent memory overload.
    n_rows = x_for_var.shape[0]
    var_vals = []
    group_size = 1000

    for start_idx in range(0, n_rows, group_size):
        end_idx = min(start_idx + group_size, n_rows)
        var_vals_group = my_gp.posterior_covariance(x_for_var[start_idx:end_idx], variance_only=True)["v(x)"]
        var_vals.extend(var_vals_group)

    assert len(var_vals) == n_rows, "Something wrong with the grouping"

    persis_info.setdefault("max_variance", []).append(np.max(var_vals))
    persis_info.setdefault("mean_variance", []).append(np.mean(var_vals))

    if test_points is not None:
        _compare_testpoints(my_gp, test_points, persis_info)

    return np.array(var_vals)


def _calculate_grid_distances(lb, ub, num_points):
    """Calculate minimum and maximum distances between points in grid"""
    num_points = [num_points] * len(lb)
    spacings = [(ub[i] - lb[i]) / (num_points[i] - 1) for i in range(len(lb))]
    min_distance = min(spacings)
    max_distance = np.sqrt(sum([(ub[i] - lb[i]) ** 2 for i in range(len(lb))]))
    return min_distance, max_distance


def _is_point_far_enough(point, eligible_points, r):
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
        if _is_point_far_enough(point, eligible_points, r):
            eligible_points.append(point)
            if len(eligible_points) == batch_size:
                break
    return np.array(eligible_points)


def persistent_gpCAM(H_in, persis_info, gen_specs, libE_info):
    """
    This generation function constructs a global surrogate of `f` values. It is
    a batched method that produces a first batch uniformly random from (lb, ub).
    On subsequent iterations, it calls an optimization method to produce the next
    batch of points. This optimization might be too slow (relative to the
    simulation evaluation time) for some use cases.

    .. seealso::
        `test_gpCAM.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_gpCAM.py>`_
    """  # noqa

    rng, batch_size, n, lb, ub, x_new, y_new, ps = _initialize_gpcAM(gen_specs["user"], libE_info, persis_info)
    ask_max_iter = gen_specs["user"].get("ask_max_iter") or 10
    test_points = _read_testpoints(gen_specs["user"])
    noise = 1e-8  # Initializes noise
    my_gp = None

    # Start with a batch of random points
    x_new = rng.uniform(lb, ub, (batch_size, n))
    H_o = np.zeros(batch_size, dtype=gen_specs["out"])
    H_o["x"] = x_new
    tag, Work, calc_in = ps.send_recv(H_o)  # Send random points for evaluation and wait

    while tag not in [STOP_TAG, PERSIS_STOP]:
        y_new = np.atleast_2d(calc_in["f"]).T
        my_gp = _update_gp(my_gp, x_new, y_new, test_points, persis_info, noise)

        # Request new points
        start = time.time()
        x_new = my_gp.ask(
            input_set=np.column_stack((lb, ub)),
            n=batch_size,
            pop_size=batch_size,
            acquisition_function="total correlation",
            max_iter=ask_max_iter,  # Larger takes longer. gpCAM default is 20.
        )["x"]
        print(f"Ask time:{time.time() - start}")

        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        H_o["x"] = x_new
        tag, Work, calc_in = ps.send_recv(H_o)

    # If final points were returned update the model
    if calc_in is not None:
        y_new = np.atleast_2d(calc_in["f"]).T
        my_gp = _update_gp(my_gp, x_new, y_new, test_points, persis_info, noise)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_gpCAM_covar(H_in, persis_info, gen_specs, libE_info):
    """
    This generation function constructs a global surrogate of `f` values.
    It is a batched method that produces a first batch uniformly random from
    (lb, ub) and on following iterations samples the GP posterior covariance
    function to find sample points.

    If gen_specs["user"]["use_grid"] is set to True, the parameter space is
    divided into a mesh of candidate points (num_points in each dimension).
    Subsequent points are chosen with maximum covariance that are at least a
    distance `r` away from each other to explore difference regions.

    If gen_specs["user"]["test_points_file"] is set to a file of evaluated
    points, then the gpCAM predications are compared at these points to assess
    model quality.

    .. seealso::
        `test_gpCAM.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_gpCAM.py>`_
    """  # noqa

    U = gen_specs["user"]
    my_gp = None
    noise = 1e-12

    test_points = _read_testpoints(U)
    rng, batch_size, n, lb, ub, x_new, y_new, ps = _initialize_gpcAM(gen_specs["user"], libE_info, persis_info)

    # Send batches until manager sends stop tag
    tag = None
    var_vals = None

    if U.get("use_grid"):
        num_points = 10
        x_for_var = _generate_mesh(lb, ub, num_points)
        r_low_init, r_high_init = _calculate_grid_distances(lb, ub, num_points)
    else:
        x_for_var = rng.uniform(lb, ub, (10 * batch_size, n))

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if x_new.shape[0] == 0:
            x_new = rng.uniform(lb, ub, (batch_size, n))
        else:
            if not U.get("use_grid"):
                x_for_var = rng.uniform(lb, ub, (10 * batch_size, n))
                x_new = x_for_var[np.argsort(var_vals)[-batch_size:]]
            else:
                r_high = r_high_init
                r_low = r_low_init
                x_new = []
                r_cand = r_high  # Let's start with a large radius and stop when we have batchsize points

                sorted_indices = np.argsort(-var_vals)
                while len(x_new) < batch_size:
                    x_new = _find_eligible_points(x_for_var, sorted_indices, r_cand, batch_size)
                    if len(x_new) < batch_size:
                        r_high = r_cand
                    r_cand = (r_high + r_low) / 2.0

        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        H_o["x"] = x_new
        tag, Work, calc_in = ps.send_recv(H_o)

        # This works with or without final_gen_send
        if calc_in is not None:
            y_new = np.atleast_2d(calc_in["f"]).T
            nan_indices = [i for i, fval in enumerate(y_new) if np.isnan(fval)]
            x_new = np.delete(x_new, nan_indices, axis=0)
            y_new = np.delete(y_new, nan_indices, axis=0)
            my_gp = _update_gp(my_gp, x_new, y_new, test_points, persis_info, noise)

            if not U.get("use_grid"):
                x_for_var = rng.uniform(lb, ub, (10 * batch_size, n))
            var_vals = _eval_var(my_gp, x_new, y_new, x_for_var, test_points, persis_info)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
