import os
import subprocess

import numpy as np

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport


def build_H0(x_f_pairs, gen_specs, noise_h_mat):
    U = gen_specs["user"]
    x0 = U["x0"]

    # This function constructs H0 to contain points to be sent back to the
    # manager to be evaluated

    n = len(x0)
    E = np.eye(n)
    nf = U["nf"]

    H0 = np.zeros(len(x_f_pairs) * nf, dtype=gen_specs["out"])
    ind = 0
    for i, j in x_f_pairs:
        for k in range(nf + 1):
            if k != nf // 2:
                H0["x"][ind] = x0 + (k - nf / 2) * noise_h_mat[i, j] * E[i]
                H0["x_ind"][ind] = i
                H0["f_ind"][ind] = j
                H0["n_ind"][ind] = k
                ind += 1

    return H0


def fd_param_finder(H, persis_info, gen_specs, libE_info):
    """
    This generation function loops through a set of suitable finite difference
    parameters for a mapping ``F`` from ``R^n`` to ``R^m``.

    .. seealso::
        `test_persistent_fd_param_finder.py` <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_fd_param_finder.py>`_ # noqa
    """
    U = gen_specs["user"]

    p = U["p"]
    x0 = U["x0"]
    nf = U["nf"]
    noise_h_mat = U["noise_h_mat"]
    inform = np.zeros_like(noise_h_mat)
    Fnoise = np.zeros_like(noise_h_mat)
    maxnoiseits = U["maxnoiseits"]
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    n = len(x0)
    Fhist0 = np.zeros((n, p, nf + 1))
    tag = None

    # # Request evaluations of the base point x0 at all p f_inds
    for i in range(n):
        for j in range(p):
            Fhist0[i, j, nf // 2] = U["f0"][j]

    x_f_pairs = np.array(np.meshgrid(range(n), range(p))).T.reshape(-1, n)
    H0 = build_H0(x_f_pairs, gen_specs, noise_h_mat)

    iters = np.ones_like(noise_h_mat)

    tag, Work, calc_in = ps.send_recv(H0)

    # Send nf points for each (x_ind, f_ind) pair
    while tag not in [STOP_TAG, PERSIS_STOP]:
        x_f_pairs = np.unique(calc_in[["x_ind", "f_ind"]])
        x_f_pairs_new = []

        # Update Fhist0
        for i, j in x_f_pairs:
            for k in range(nf + 1):
                if k != nf / 2:
                    logical_conds = (calc_in["x_ind"] == i, calc_in["f_ind"] == j, calc_in["n_ind"] == k)
                    Fhist0[i, j, k] = calc_in["f_val"][np.logical_and.reduce(logical_conds)][0]

            cmd = [
                "octave",
                "--no-window-system",
                "--eval",
                "F=[" + " ".join([f"{x:18.18f}" for x in Fhist0[i, j, : nf + 1]]) + "];"
                "nf=" + str(nf) + "';"
                "[fnoise, ~, inform] = ECnoise(nf+1, F);"
                "dlmwrite('fnoise.out', fnoise, 'delimiter', ' ', 'precision', 16);"
                "dlmwrite('inform.out', inform, 'delimiter', ' ', 'precision', 16);"
                "exit",
            ]

            p = subprocess.call(cmd, shell=False, stdout=subprocess.DEVNULL)

            inform[i, j] = np.loadtxt("inform.out")

            if inform[i, j] >= 2:
                # Mark as needing more points for this noise_h_mat value
                if iters[i, j] < maxnoiseits:
                    iters[i, j] += 1
                    x_f_pairs_new.append((i, j))

                    if inform[i, j] == 3:
                        noise_h_mat[i, j] = noise_h_mat[i, j] / 100
                    else:
                        noise_h_mat[i, j] = noise_h_mat[i, j] * 100
            else:
                # We have successfully identified the Fnoise
                Fnoise[i, j] = np.loadtxt("fnoise.out")

            os.remove("inform.out")
            os.remove("fnoise.out")

        if np.all(inform == 1):
            break

        H0 = build_H0(x_f_pairs_new, gen_specs, noise_h_mat)
        tag, Work, calc_in = ps.send_recv(H0)

    persis_info["Fnoise"] = Fnoise
    return H0, persis_info, FINISHED_PERSISTENT_GEN_TAG
