import numpy as np
from libensemble.gen_funcs.persistent_aposmm import (
    initialize_APOSMM,
    update_history_dist,
    decide_where_to_start_localopt,
)
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func
from libensemble.tests.regression_tests.support import six_hump_camel_minima as known_minima


def setup_history_and_find_rk(n_s, num_to_start, lb, ub, f_vals, x_points):
    """
    Populate the history array H with n_s points and bisect over r_k to find a value
    producing num_to_start local optimization start points using decide_where_to_start_localopt.

    Parameters:
    - n_s (int): Number of initial sample points.
    - num_to_start (int): Desired number of starting points for local optimization.
    - lb, ub (np.ndarray): Lower and upper bounds of the domain.
    - f_vals (np.ndarray): Function values at each x_point.
    - x_points (np.ndarray): n_s x d array of sample points.

    Returns:
    - H (np structured array): Updated history array.
    - rk_final (float): Value of r_k yielding num_to_start local opt starts.
    - inds_to_start (list): Indices in H to start local optimization.
    """

    assert x_points.shape[0] == n_s
    assert f_vals.shape[0] == n_s

    n = x_points.shape[1]

    H = np.zeros(
        n_s,
        dtype=[
            ("sim_id", int),
            ("x", float, n),
            ("x_on_cube", float, n),
            ("f", float),
            ("local_pt", bool),
            ("sim_ended", bool),
        ],
    )

    # Setup history
    for i in range(n_s):
        H[i]["x"] = x_points[i]
        H[i]["sim_id"] = i
        H[i]["x_on_cube"] = (x_points[i] - lb) / (ub - lb)
        H[i]["f"] = f_vals[i]
        H[i]["local_pt"] = False
        H[i]["sim_ended"] = True  # Ensure point is considered by distance function

    user_specs = {"lb": lb, "ub": ub, "initial_sample_size": n_s, "localopt_method": None}
    local_H = initialize_APOSMM(H, user_specs, {"comm": None})[-1]
    local_H = local_H[:n_s]  # Use only the required number of entries

    update_history_dist(local_H, n)

    # Search to find r_k that yields exactly num_to_start start points
    r_low, r_high = 1e-5, 2.0  # Conservative initial bounds
    rk_final = None
    tol = 1e-5

    while r_high - r_low > tol:
        r_mid = (r_low + r_high) / 2.0
        inds_to_start = decide_where_to_start_localopt(local_H, n, n_s, r_mid)

        if len(inds_to_start) < num_to_start:
            r_high = r_mid
        elif len(inds_to_start) > num_to_start:
            r_low = r_mid
        else:
            rk_final = r_mid
            break

    # Final decision (in case we didn't hit num_to_start exactly)
    if rk_final is None:
        rk_final = (r_low + r_high) / 2.0
        inds_to_start = decide_where_to_start_localopt(local_H, n, n_s, rk_final)

    return local_H, rk_final, inds_to_start


if __name__ == "__main__":

    # Define domain bounds
    lb = np.array([-3.0, -2.0])
    ub = np.array([3.0, 2.0])
    dim = len(lb)

    # Number of sample points and desired number of start points
    num_samples = 1000
    num_to_start = 6

    # Generate random sample points uniformly in the [lb,ub] box
    x_points = lb + (ub - lb) * np.random.uniform(size=(num_samples, dim))

    # Evaluate six-hump camel function at each point
    f_vals = np.array([six_hump_camel_func(x) for x in x_points])

    # Call the history setup and bisection function
    H, rk_final, inds_to_start = setup_history_and_find_rk(num_samples, num_to_start, lb, ub, f_vals, x_points)

    assert len(inds_to_start) == num_to_start, f"Found {len(inds_to_start)} starting points instead of {num_to_start}"

    starting_pts = H["x"][inds_to_start]
    sorted_starting = starting_pts[np.lexsort(starting_pts.T[::-1])]
    sorted_known = known_minima[np.lexsort(known_minima.T[::-1])]

    print(
        f"For this problem, we know the minima.\n"
        f"The chosen starting points:\n{sorted_starting}\n"
        f"should be close to the known minima:\n{sorted_known}"
    )

    # Output results
    print(f"Chosen r_k: {rk_final:.6f}")
    print(f"Indices to start local optimization (num_to_start={num_to_start}): {inds_to_start}")
