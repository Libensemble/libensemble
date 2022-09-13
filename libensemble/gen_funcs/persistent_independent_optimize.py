"""
@Author: Created by Caleb Ju at Argonne National Labs as Given's associate,
         Summer 2021.
"""
import numpy as np
import scipy.optimize as sciopt

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.consensus_subroutines import print_final_score, get_grad, get_func


def independent_optimize(H, persis_info, gen_specs, libE_info):
    """Uses scipy.optimize to solve objective function"""
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]

    eps = persis_info["params"]["eps"]
    f_i_idxs = persis_info["f_i_idxs"]

    def _f(x):
        tag, f_val = get_func(x, f_i_idxs, gen_specs, libE_info)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return np.inf
        return f_val

    def _df(x):
        tag, gradf_val = get_grad(x, f_i_idxs, gen_specs, libE_info)
        if tag in [STOP_TAG, PERSIS_STOP]:
            # Trick optimizer to think we found minimum
            return np.zeros(len(x))
        return gradf_val

    while 1:
        x0 = persis_info["rand_stream"].uniform(low=lb, high=ub)

        res = sciopt.minimize(
            _f,
            x0,
            jac=_df,
            method="BFGS",
            tol=eps,
            options={
                "gtol": eps,
                "norm": np.inf,
                "maxiter": None,
            },
        )
        print_final_score(res.x, f_i_idxs, gen_specs, libE_info)

        start_pt, end_pt = f_i_idxs[0], f_i_idxs[-1]
        print(f"[Worker {persis_info['worker_num']}]: x={res.x[2 * start_pt:2 * end_pt]}", flush=True)
        """
        try:
           res = sciopt.minimize(_f, x0, jac=_df, method="BFGS", tol=eps,
                                  options={'gtol': eps, 'norm': np.inf, 'maxiter': None})
           print_final_score(res.x, f_i_idxs, gen_specs, libE_info)

        except:
           print('hi', flush=True)
           return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
        """

        return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
