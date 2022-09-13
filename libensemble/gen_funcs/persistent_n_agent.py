"""
@Author: Created by Caleb Ju at Argonne National Labs as Given's associate,
         Summer 2021.
@About: Based on n-agent with gradient tracking:
        https://ieeexplore.ieee.org/abstract/document/9199106/
"""
import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.consensus_subroutines import print_final_score, get_grad, get_neighbor_vals


def n_agent(H, persis_info, gen_specs, libE_info):
    """Gradient sliding. Coordinates with alloc to do local and distributed
    (i.e., gradient of consensus step) calculations.
    """
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    n = len(ub)

    # start with random x0
    x0 = persis_info["rand_stream"].uniform(low=lb, high=ub, size=(n,))
    x_k = x0

    L = persis_info["params"]["L"]
    eps = persis_info["params"]["eps"]
    rho = persis_info["params"]["rho"]
    N_const = persis_info["params"]["N_const"]
    step_const = persis_info["params"]["step_const"]

    N = int(N_const / eps + 1)
    eta = step_const * 1.0 / L * min(1 / 6, (1 - rho**2) ** 2 / (4 * rho**2 * (3 + 4 * rho**2)))

    f_i_idxs = persis_info["f_i_idxs"]
    A_i_data = persis_info["A_i_data"]
    A_i_gen_ids = persis_info["A_i_gen_ids"]
    local_gen_id = persis_info["worker_num"]

    # sort A_i to be increasing gen_id
    _perm_ids = np.argsort(A_i_gen_ids)
    A_weights = A_i_data[_perm_ids]
    A_i_gen_ids = A_i_gen_ids[_perm_ids]
    A_i_gen_ids_no_local = np.delete(A_i_gen_ids, np.where(A_i_gen_ids == local_gen_id)[0][0])

    prev_s_is = np.zeros((len(A_i_data), n), dtype=float)
    prev_gradf_is = np.zeros((len(A_i_data), n), dtype=float)

    if local_gen_id == 1:
        print(f"[{0}%]: ", flush=True, end="")
    print_final_score(x_k, f_i_idxs, gen_specs, libE_info)
    percent = 0.1

    for k in range(N):
        tag, gradf = get_grad(x_k, f_i_idxs, gen_specs, libE_info)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        tag, neighbor_gradf_is = get_neighbor_vals(gradf, local_gen_id, A_i_gen_ids_no_local, gen_specs, libE_info)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        U = prev_s_is + neighbor_gradf_is - prev_gradf_is
        # takes linear combination as described by equation (9)
        s = np.dot(U.T, A_weights)

        tag, neighbor_x_is = get_neighbor_vals(x_k, local_gen_id, A_i_gen_ids_no_local, gen_specs, libE_info)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
        tag, neighbor_s_is = get_neighbor_vals(s, local_gen_id, A_i_gen_ids_no_local, gen_specs, libE_info)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        V = neighbor_x_is - eta * neighbor_s_is
        # takes linear combination as described by equation (10)
        next_x_k = np.dot(V.T, A_weights)

        x_k = next_x_k

        # Save data to avoid more communication
        prev_s_is = neighbor_s_is
        prev_gradf_is = neighbor_gradf_is

        if (k + 1) / N >= percent:
            if local_gen_id == 1:
                print(f"[{int(percent * 100)}%]: ", flush=True, end="")
            percent += 0.1
            print_final_score(x_k, f_i_idxs, gen_specs, libE_info)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
