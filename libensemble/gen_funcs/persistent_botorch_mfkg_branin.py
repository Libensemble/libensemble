"""
This file defines the persistent generator function for multi-fidelity Bayesian
optimization using BoTorch's Multi-Fidelity Knowledge Gradient (MFKG) acquisition function.

This gen_f is meant to be used with the alloc_f function `only_persistent_gens`.
"""

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
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = ["persistent_botorch_mfkg"]

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Torch settings
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Specify bounds
dim = 3
bounds = torch.tensor([[0.0] * dim, [1.0] * dim], **tkwargs)

# Specify target fidelity
target_fidelities = {2: 1.0}

# Specify cost model
cost_model = AffineFidelityCostModel(fidelity_weights={2: 0.9}, fixed_cost=0.1)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)


# Custom function to project posterior to target fidelity (defer to default)
def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)


# Wrapper function for compatibility with existing code
def problem(X, ps, gen_specs):
    """
    Wrapper to convert tensor input to numpy and send to libE for evaluation.
    
    Args:
        X: tensor of shape (n, 3) where columns are [x0, x1, fidelity]
        ps: PersistentSupport object for communication
        gen_specs: Generator specifications
    
    Returns:
        tensor of shape (n,) with objective values
    """
    # Send points to be evaluated
    X_np = X.cpu().numpy()
    H_o = np.zeros(len(X), dtype=gen_specs["out"])
    H_o["x"] = X_np[:, :2]
    H_o["fidelity"] = X_np[:, 2]
    
    tag, Work, calc_in = ps.send_recv(H_o)
    
    # Convert results back to tensor
    if calc_in is None or len(calc_in) == 0:
        return None, tag
    
    train_obj = torch.tensor(calc_in["f"], **tkwargs).unsqueeze(-1)
    
    return train_obj, tag


# Function to generate training data
def generate_initial_data(n, ps, gen_specs):  # Jeff: Initial sample size is twice this value of n
    train_x = torch.rand(n, 2, **tkwargs)
    train_lf = torch.zeros(n, 1)
    train_hf = torch.ones(n, 1)
    train_x_full_lf = torch.cat((train_x, train_lf), dim=1)
    train_x_full_hf = torch.cat((train_x, train_hf), dim=1)
    train_x_full = torch.cat((train_x_full_lf, train_x_full_hf), dim=0)
    train_obj, tag = problem(train_x_full, ps, gen_specs)
    return train_x_full, train_obj, tag


# Function to initialize a botorch model
def initialize_model(train_x, train_obj):
    model = SingleTaskMultiFidelityGP(train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelities=[2])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# Multifidelity Knowledge Gradient acquisition function
def get_mfkg(model):

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[1],
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,  # Jeff: Don't adjust this for some reason
        num_restarts=1,  # Jeff: I decreased this to make libE development faster
        raw_samples=10,  # Jeff: I decreased this to make libE development faster
        options={"batch_limit": 10, "maxiter": 10},  # Jeff: I decreased this to make libE development faster
    )

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


# Optimization step
def optimize_mfkg_and_get_observation(mfkg_acqf, q, ps, gen_specs):
    # Generate new candidates
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{2: 0.0}, {2: 1.0}],
        q=q,  # Jeff: This is the number of new samples to make
        num_restarts=1,  # Jeff: I decreased this to make libE development faster
        raw_samples=10,  # Jeff: I decreased this to make libE development faster
        options={"batch_limit": 10, "maxiter": 10},  # Jeff: I decreased this to make libE development faster
    )

    # Observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj, tag = problem(new_x, ps, gen_specs)
    return new_x, new_obj, cost, tag


# Function to perform a single iteration
def do_iteration(train_x, train_obj, q, ps, gen_specs):
    mll, model = initialize_model(train_x, train_obj)
    fit_gpytorch_mll(mll)
    mfkg_acqf = get_mfkg(model)
    new_x, new_obj, _, tag = optimize_mfkg_and_get_observation(mfkg_acqf, q, ps, gen_specs)
    
    if new_obj is None:
        return model, train_x, train_obj, tag
    
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])  # Jeff: This is where the "sim" evaluation happens, and needs to be communicated back to the manager

    return model, train_x, train_obj, tag


def persistent_botorch_mfkg(H, persis_info, gen_specs, libE_info):
    """
    Persistent generator function for multi-fidelity Bayesian optimization using BoTorch's MFKG.
    """
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Extract user parameters
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]
    n_init_samples = gen_specs["user"]["n_init_samples"]
    q = gen_specs["user"]["q"]

    # ## Perform Multifidelity Bayesian Optimization
    # Generate initial data
    train_x, train_obj, tag = generate_initial_data(n_init_samples, ps, gen_specs)

    # Step
    while tag not in [STOP_TAG, PERSIS_STOP]:
        model, train_x, train_obj, tag = do_iteration(train_x, train_obj, q, ps, gen_specs)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
