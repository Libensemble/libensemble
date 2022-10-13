"""
@Author: Created by Caleb Ju at Argonne National Labs as Given's associate,
         Summer 2021.
@About: Based on communication sliding primal-dual method:
        https://link.springer.com/article/10.1007/s10107-018-1355-4
"""
import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.consensus_subroutines import (
    print_final_score,
    get_grad,
    get_consensus_gradient,
    get_grad_locally,
)


def opt_slide(H, persis_info, gen_specs, libE_info):
    """Gradient sliding. Coordinates with alloc to do local and distributed
    (i.e., gradient of consensus step) calculations.
    """
    # Send batches until manager sends stop tag
    tag = None
    local_gen_id = persis_info["worker_num"]

    # each gen has unique internal id
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    n = len(ub)

    f_i_idxs = persis_info["f_i_idxs"]

    # start with random x_0
    x_0 = persis_info["rand_stream"].uniform(low=lb, high=ub, size=(n,))
    # ===== NOTATION =====
    # x_hk == \hat{x}_k
    # x_tk == \tilde{x}_k
    # x_uk == \underline{x}_k
    # prev_x_k == x_{k-1}
    # prevprev_x_k = x_{k-2}
    # ====================
    prev_x_k = x_0.copy()
    prev_x_uk = x_0.copy()
    prev_x_hk = x_0.copy()
    prev_z_k = np.zeros(len(x_0), dtype=float)
    prevprev_x_k = x_0.copy()
    prev_penult_k = x_0.copy()

    mu = persis_info["params"]["mu"]
    L = persis_info["params"]["L"]
    A_norm = persis_info["params"]["A_norm"]
    Vx_0x = persis_info["params"]["Vx_0x"]
    eps = persis_info["params"]["eps"]
    f_i_eval = persis_info["params"].get("f_i_eval", None)
    df_i_eval = persis_info["params"].get("df_i_eval", None)

    R = 1.0 / (4 * (Vx_0x) ** 0.5)
    N = int(4 * (L * Vx_0x / eps) ** 0.5 + 1)

    weighted_x_hk_sum = np.zeros(len(x_0), dtype=float)
    b_k_sum = 0

    prev_b_k = 0
    prev_T_k = 0

    if local_gen_id == 1:
        print(f"[{0}%]: ", flush=True, end="")
    print_final_score(prev_x_k, f_i_idxs, gen_specs, libE_info)
    percent = 0.1

    for k in range(1, N + 1):
        tau_k = (k - 1) / 2
        lam_k = (k - 1) / k
        b_k = k
        p_k = 2 * L / k
        T_k = int(k * R * A_norm / L + 1)

        x_tk = prev_x_k + lam_k * (prev_x_hk - prevprev_x_k)
        x_uk = (x_tk + tau_k * prev_x_uk) / (1 + tau_k)

        if f_i_eval is not None:
            y_k = get_grad_locally(x_uk, f_i_idxs, df_i_eval)
        else:
            tag, y_k = get_grad(x_uk, f_i_idxs, gen_specs, libE_info)

        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        settings = {
            "T_k": T_k,
            "b_k": k,
            "p_k": p_k,
            "mu": mu,
            "L": L,
            "R": R,
            "k": k,
            "prev_b_k": prev_b_k,
            "prev_T_k": prev_T_k,
        }

        [tag, x_k, x_k_1, z_k, x_hk] = primaldual_slide(
            y_k, prev_x_k, prev_penult_k, prev_z_k, settings, gen_specs, libE_info
        )
        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        prevprev_x_k = prev_x_k
        prev_x_k = x_k
        prev_x_hk = x_hk
        prev_penult_k = x_k_1  # penultimate x_k^{(i)}
        prev_z_k = z_k
        prev_b_k = b_k
        prev_T_k = T_k
        prev_x_uk = x_uk

        weighted_x_hk_sum += b_k * x_hk
        b_k_sum += b_k

        if k / N >= percent:
            curr_x_star = 1.0 / b_k_sum * weighted_x_hk_sum
            if local_gen_id == 1:
                print(f"[{int(percent * 100)}%]: ", flush=True, end="")
            percent += 0.1
            print_final_score(curr_x_star, f_i_idxs, gen_specs, libE_info)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def primaldual_slide(y_k, x_curr, x_prev, z_t, settings, gen_specs, libE_info):
    tag = None

    # define params
    T_k = settings["T_k"]
    b_k = settings["b_k"]
    p_k = settings["p_k"]
    mu = settings["mu"]
    L = settings["L"]
    R = settings["R"]
    k = settings["k"]
    prev_b_k = settings["prev_b_k"]
    prev_T_k = settings["prev_T_k"]

    x_k_1 = x_curr.copy()
    xsum = np.zeros(len(x_curr), dtype=float)

    for t in range(1, T_k + 1):
        # define per-iter params
        eta_t = (p_k + mu) * (t - 1) + p_k * T_k
        q_t = L * T_k / (2 * b_k * R**2)
        if k >= 2 and t == 1:
            a_t = prev_b_k * T_k / (b_k * prev_T_k)
        else:
            a_t = 1

        u_t = x_curr + a_t * (x_curr - x_prev)

        # compute first argmin
        tag, Lu_t = get_consensus_gradient(u_t, gen_specs, libE_info)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return [tag, None, None, None, None]

        z_t = z_t + (1.0 / q_t) * Lu_t

        # computes second argmin
        tag, Lz_t = get_consensus_gradient(z_t, gen_specs, libE_info)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return [tag, None, None, None, None]

        x_next = (eta_t * x_curr) + (p_k * x_k_1) - (y_k + Lz_t)
        x_next /= eta_t + p_k

        x_prev = x_curr
        x_curr = x_next

        xsum += x_curr
        # zsum += z_t

    x_k = x_curr
    x_k_1 = x_prev
    z_k = z_t
    x_hk = xsum / T_k

    return [tag, x_k, x_k_1, z_k, x_hk]
