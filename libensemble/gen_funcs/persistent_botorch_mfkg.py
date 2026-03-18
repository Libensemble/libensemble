import os
import warnings
from typing import Tuple

import numpy as np
import torch

# BoTorch / GPyTorch
from botorch import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

warnings.filterwarnings("ignore")


def _to_torch(arr, device, dtype):
    return torch.as_tensor(np.asarray(arr), device=device, dtype=dtype)


def persistent_mfkg_gen_f(H, persis_info, gen_specs, libE_info):
    """libEnsemble persistent generator using BoTorch Multi-Fidelity KG (MF-KG).

    Expects gen_specs['user'] to contain:
      - lb, ub: lists of bounds
      - name_hifi, name_lofi: libE task names for hi/lo fidelity evals
      - n_init_hifi, n_init_lofi: initial design sizes
      - n_opt_hifi, n_opt_lofi: number of points to propose per iteration (total = their sum)
      - (optional) maximize: bool (default False). If False, we minimize f. MF-KG maximizes, so we model y = sign * f with sign = +1 if maximize else -1.

    The last dimension of the BoTorch input is fidelity s in {0,1}, appended to x:
        X_full = [x0, x1, ..., x_{d-1}, s]
    """
    # Extract bounds of the parameter space, and batch size
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]

    # Get task names.
    hifi_task = gen_specs["user"]["name_hifi"]
    lofi_task = gen_specs["user"]["name_lofi"]

    # Number of points to generate initially and during optimization.
    n_init_hifi = gen_specs["user"]["n_init_hifi"]
    n_init_lofi = gen_specs["user"]["n_init_lofi"]
    n_opt_hifi = gen_specs["user"]["n_opt_hifi"]
    n_opt_lofi = gen_specs["user"]["n_opt_lofi"]

    maximize = bool(gen_specs["user"].get("maximize", False))
    sign = 1.0 if maximize else -1.0  # MF-KG maximizes; if we need to minimize, flip sign.

    # Torch device / dtype
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    bounds = torch.stack((_to_torch(lb.tolist() + [0.0], **tkwargs), _to_torch(ub.tolist() + [1.0], **tkwargs)))

    # Target fidelity is s=1 (index d)
    fidelity_index = 1
    target_fidelities = {fidelity_index: 1.0}

    # Cost model / utility (tune weights/fixed_cost as desired)
    cost_model = AffineFidelityCostModel(fidelity_weights={fidelity_index: 0.9}, fixed_cost=0.1)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    def project(X: torch.Tensor) -> torch.Tensor:
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    def _initialize_model(train_x: torch.Tensor, train_y: torch.Tensor):
        model = SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=1),
            data_fidelities=[fidelity_index],
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def _current_value(model) -> torch.Tensor:
        """Posterior mean at target fidelity, maximized over x (with s fixed to 1)."""
        pm = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=d + 1,
            columns=[fidelity_index],
            values=[1.0],
        )
        # Optimize over the d design dims (no fidelity dim here)
        x_rec, _ = optimize_acqf(
            acq_function=pm,
            bounds=bounds[:, :d],
            q=1,
            num_restarts=8,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
        )
        # Build full X including fidelity=1
        X_full = pm._construct_X_full(x_rec)
        with torch.no_grad():
            val = model.posterior(X_full).mean
        return val

    def _get_mfkg(model):
        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=128,
            current_value=_current_value(model),
            cost_aware_utility=cost_aware_utility,
            project=project,
        )

    # Helper to submit a batch of points of one fidelity to libE and collect y
    def _eval_with_libE(X_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # X_full shape: (m, d+1), last column is fidelity s in {0,1}
        m = X_full.shape[0]
        H_o = np.zeros(m, dtype=gen_specs["out"])
        for i in range(m):
            x = X_full[i, :d].detach().cpu().numpy()
            s = float(X_full[i, fidelity_index].item())
            H_o["x"][i] = x
            H_o["resource_sets"][i] = 1
            H_o["task"][i] = name_hifi if s >= 0.5 else name_lofi

        tag, _, calc_in = ps.send_recv(H_o)
        if tag in (STOP_TAG, PERSIS_STOP):
            return tag, None  # signal to caller

        fvals = calc_in["f"][:m]
        y = _to_torch(sign * np.asarray(fvals).reshape(-1, 1), **tkwargs)  # y = sign * f
        return None, y

    # Persistent support
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # === Initialization: random designs at each fidelity ===
    rng = np.random.default_rng(int(persis_info.get("seed", 0)))

    def _rand_in_bounds(n, s_val):
        U = rng.random((n, d))
        X = lb + U * (ub - lb)
        S = np.full((n, 1), float(s_val))
        return _to_torch(np.hstack([X, S]), **tkwargs)

    train_x_list = []
    train_y_list = []

    # Low-fidelity init
    if n_init_lofi > 0:
        X_lo = _rand_in_bounds(n_init_lofi, 0.0)
        tag, y_lo = _eval_with_libE(X_lo)
        if tag in (STOP_TAG, PERSIS_STOP):
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
        train_x_list.append(X_lo)
        train_y_list.append(y_lo)

    # High-fidelity init
    if n_init_hifi > 0:
        X_hi = _rand_in_bounds(n_init_hifi, 1.0)
        tag, y_hi = _eval_with_libE(X_hi)
        if tag in (STOP_TAG, PERSIS_STOP):
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
        train_x_list.append(X_hi)
        train_y_list.append(y_hi)

    train_x = torch.cat(train_x_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # === BO loop ===
    iter_idx = 0
    while True:
        # Fit / refit GP
        mll, model = _initialize_model(train_x, train_y)
        fit_gpytorch_mll(mll)

        # Build MF-KG acqf
        mfkg = _get_mfkg(model)

        # We’ll generate q_per_iter new points, allowing s to be 0 or 1.
        # optimize_acqf_mixed picks the fidelity via fixed_features_list entries.
        # Here we give the optimizer the choice: try s=0 and s=1 separately and take best.
        new_x_all = []
        new_y_all = []

        for _ in range(q_per_iter):
            candidates, _ = optimize_acqf_mixed(
                acq_function=mfkg,
                bounds=bounds,
                fixed_features_list=[{fidelity_index: 0.0}, {fidelity_index: 1.0}],
                q=1,
                num_restarts=6,
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 200},
            )
            # Evaluate this single point
            tag, y_new = _eval_with_libE(candidates)
            if tag in (STOP_TAG, PERSIS_STOP):
                return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

            # Update training data (greedy refit next iteration)
            train_x = torch.cat([train_x, candidates], dim=0)
            train_y = torch.cat([train_y, y_new], dim=0)
            new_x_all.append(candidates)
            new_y_all.append(y_new)

        # (Optional) save snapshots for debugging
        if iter_idx == 0 and not os.path.exists("mfkg_model_history"):
            os.mkdir("mfkg_model_history")
        torch.save(
            {"train_x": train_x.cpu(), "train_y": train_y.cpu()},
            os.path.join("mfkg_model_history", f"snapshot_{iter_idx:04d}.pt"),
        )
        iter_idx += 1

    # Unreachable
    # return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
