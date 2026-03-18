import os
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from botorch import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

warnings.filterwarnings("ignore")


def _to_torch(arr, *, device, dtype):
    return torch.as_tensor(np.asarray(arr), device=device, dtype=dtype)


def persistent_mfkg_gen_f(H, persis_info, gen_specs, libE_info):
    """libEnsemble persistent generator using BoTorch Multi-Fidelity KG (MF-KG).

    Required gen_specs['user'] entries:
      - lb, ub
      - name_hifi, name_lofi
      - n_init_hifi, n_init_lofi
      - n_opt_hifi, n_opt_lofi

    Optional:
      - maximize (default False)
      - gen_batch_size (default: n_opt_hifi + n_opt_lofi)
      - debug (default False)
      - debug_q (default 1 when debug=True)
      - save_dir (default "mfkg_model_history")
      - num_fantasies, kg_raw_samples, kg_num_restarts,
        cv_raw_samples, cv_num_restarts, opt_maxiter
        (optional overrides for non-debug mode)

    We append a fidelity variable s to x:
        X_full = [x0, ..., x_{d-1}, s]
    with s in {0.0, 1.0}, where s=1.0 is the target (high) fidelity.
    """
    user = gen_specs["user"]

    # Convert bounds to numpy arrays so arithmetic and shape logic are reliable.
    lb = np.asarray(user["lb"], dtype=float)
    ub = np.asarray(user["ub"], dtype=float)
    if lb.shape != ub.shape:
        raise ValueError(f"lb and ub must have the same shape, got {lb.shape} and {ub.shape}")
    if lb.ndim != 1:
        raise ValueError("lb and ub must be 1-D arrays/lists")
    if np.any(ub <= lb):
        raise ValueError("All entries of ub must be strictly greater than lb")

    d = lb.size
    fidelity_index = d  # fidelity is appended as the last coordinate

    hifi_task = user["name_hifi"]
    lofi_task = user["name_lofi"]

    n_init_hifi = int(user["n_init_hifi"])
    n_init_lofi = int(user["n_init_lofi"])
    n_opt_hifi = int(user["n_opt_hifi"])
    n_opt_lofi = int(user["n_opt_lofi"])
    requested_q = n_opt_hifi + n_opt_lofi
    if requested_q <= 0:
        raise ValueError("n_opt_hifi + n_opt_lofi must be positive")

    maximize = bool(user.get("maximize", False))
    sign = 1.0 if maximize else -1.0  # MF-KG maximizes; if we need to minimize, flip sign.

    debug = bool(user.get("debug", False))
    gen_batch_size = int(user.get("gen_batch_size", requested_q))

    if debug:
        q_per_iter = int(user.get("debug_q", 1))
        num_fantasies = int(user.get("num_fantasies", 4))
        kg_raw_samples = int(user.get("kg_raw_samples", 16))
        kg_num_restarts = int(user.get("kg_num_restarts", 2))
        cv_raw_samples = int(user.get("cv_raw_samples", 16))
        cv_num_restarts = int(user.get("cv_num_restarts", 2))
        opt_maxiter = int(user.get("opt_maxiter", 50))
    else:
        q_per_iter = min(requested_q, gen_batch_size)
        num_fantasies = int(user.get("num_fantasies", 128))
        kg_raw_samples = int(user.get("kg_raw_samples", 256))
        kg_num_restarts = int(user.get("kg_num_restarts", 6))
        cv_raw_samples = int(user.get("cv_raw_samples", 256))
        cv_num_restarts = int(user.get("cv_num_restarts", 8))
        opt_maxiter = int(user.get("opt_maxiter", 200))

    if q_per_iter <= 0:
        raise ValueError("q_per_iter must be positive")
    q_per_iter = min(q_per_iter, gen_batch_size)

    # Torch device / dtype
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    bounds = torch.stack((_to_torch(np.concatenate([lb, [0.0]]), **tkwargs),_to_torch(np.concatenate([ub, [1.0]]), **tkwargs)))

    target_fidelities = {fidelity_index: 1.0}

    # Example affine cost model: cost = fixed_cost + weight * s
    cost_model = AffineFidelityCostModel(
        fidelity_weights={fidelity_index: 0.5},
        fixed_cost=0.5,
    )
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    def project(X: torch.Tensor) -> torch.Tensor:
        return project_to_target_fidelity(
            X=X,
            target_fidelities=target_fidelities,
            d=d + 1,
        )

    def _initialize_model(train_x: torch.Tensor, train_y: torch.Tensor):
        model = SingleTaskMultiFidelityGP(
            train_X=train_x,
            train_Y=train_y,
            outcome_transform=Standardize(m=1),
            data_fidelities=[fidelity_index],
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def _current_value(model) -> torch.Tensor:
        """Max posterior mean at the target fidelity."""
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=d + 1,
            columns=[fidelity_index],
            values=[1.0],
        )
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds[:, :-1],  # optimize only over the design variables
            q=1,
            num_restarts=cv_num_restarts,
            raw_samples=cv_raw_samples,
            options={"batch_limit": 5, "maxiter": opt_maxiter},
        )
        return current_value

    def _get_mfkg(model):
        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=num_fantasies,
            current_value=_current_value(model),
            cost_aware_utility=cost_aware_utility,
            project=project,
        )

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    def _eval_with_libE(X_full: torch.Tensor) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Submit a batch to libEnsemble and return observed y values."""
        m = X_full.shape[0]
        H_o = np.zeros(m, dtype=gen_specs["out"])

        for i in range(m):
            x = X_full[i, :d].detach().cpu().numpy()
            s = float(X_full[i, fidelity_index].item())
            H_o["x"][i] = x
            H_o["resource_sets"][i] = 1
            H_o["task"][i] = hifi_task if s >= 0.5 else lofi_task

        tag, _, calc_in = ps.send_recv(H_o)
        if tag in (STOP_TAG, PERSIS_STOP):
            return tag, None

        fvals = np.asarray(calc_in["f"][:m], dtype=float).reshape(-1, 1)
        y = _to_torch(sign * fvals, **tkwargs)
        return None, y

    # === Initialization: random designs at each fidelity ===
    rng = np.random.default_rng(int(persis_info.get("seed", 0)))

    def _rand_in_bounds(n: int, s_val: float) -> torch.Tensor:
        U = rng.random((n, d))
        X = lb + U * (ub - lb)
        S = np.full((n, 1), float(s_val))
        return _to_torch(np.hstack([X, S]), **tkwargs)

    train_x_list = []
    train_y_list = []

    # Low-fidelity init
    if n_init_lofi > 0:
        print(f"[gen] initial low-fidelity batch: {n_init_lofi}")
        X_lo = _rand_in_bounds(n_init_lofi, 0.0)
        tag, y_lo = _eval_with_libE(X_lo)
        if tag in (STOP_TAG, PERSIS_STOP):
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
        train_x_list.append(X_lo)
        train_y_list.append(y_lo)

    # High-fidelity init
    if n_init_hifi > 0:
        print(f"[gen] initial high-fidelity batch: {n_init_hifi}")
        X_hi = _rand_in_bounds(n_init_hifi, 1.0)
        tag, y_hi = _eval_with_libE(X_hi)
        if tag in (STOP_TAG, PERSIS_STOP):
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
        train_x_list.append(X_hi)
        train_y_list.append(y_hi)

    if not train_x_list:
        raise ValueError("At least one of n_init_lofi or n_init_hifi must be positive")

    train_x = torch.cat(train_x_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # === BO loop ===
    iter_idx = 0
    save_dir = user.get("save_dir", "mfkg_model_history")
    os.makedirs(save_dir, exist_ok=True)

    print(
        "[gen] starting BO loop with "
        f"debug={debug}, q_per_iter={q_per_iter}, "
        f"num_fantasies={num_fantasies}, "
        f"kg_num_restarts={kg_num_restarts}, kg_raw_samples={kg_raw_samples}, "
        f"cv_num_restarts={cv_num_restarts}, cv_raw_samples={cv_raw_samples}, "
        f"opt_maxiter={opt_maxiter}"
    )

    while True:
        print(f"[gen] iter {iter_idx}: fitting model on {train_x.shape[0]} points")
        mll, model = _initialize_model(train_x, train_y)
        fit_gpytorch_mll(mll)

        # Save immediately after fit so hangs in acquisition optimization are visible.
        torch.save(
            {"train_x": train_x.cpu(), "train_y": train_y.cpu()},
            os.path.join(save_dir, f"preopt_{iter_idx:04d}.pt"),
        )

        print(f"[gen] iter {iter_idx}: building MF-KG")
        mfkg = _get_mfkg(model)

        print(f"[gen] iter {iter_idx}: generating KG initial conditions")
        X_init = gen_one_shot_kg_initial_conditions(
            acq_function=mfkg,
            bounds=bounds,
            q=q_per_iter,
            num_restarts=kg_num_restarts,
            raw_samples=kg_raw_samples,
        )

        print(f"[gen] iter {iter_idx}: optimizing MF-KG with q={q_per_iter}")
        candidates, _ = optimize_acqf_mixed(
            acq_function=mfkg,
            bounds=bounds,
            fixed_features_list=[
                {fidelity_index: 0.0},
                {fidelity_index: 1.0},
            ],
            q=q_per_iter,
            num_restarts=kg_num_restarts,
            raw_samples=kg_raw_samples,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": opt_maxiter},
        )
        print(f"[gen] iter {iter_idx}: done optimizing")

        tag, y_new = _eval_with_libE(candidates)
        if tag in (STOP_TAG, PERSIS_STOP):
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        train_x = torch.cat([train_x, candidates], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)

        torch.save(
            {"train_x": train_x.cpu(), "train_y": train_y.cpu()},
            os.path.join(save_dir, f"snapshot_{iter_idx:04d}.pt"),
        )
        print(f"[gen] iter {iter_idx}: saved snapshot and continuing")

        iter_idx += 1
