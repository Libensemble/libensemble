"""
Generator class for multi-fidelity Bayesian optimization using BoTorch's
Multi-Fidelity Knowledge Gradient (MFKG) acquisition function.

Conforms to the gest-api ``Generator`` interface (``suggest``/``ingest``).
"""

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
from gest_api import Generator
from gest_api.vocs import VOCS
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

__all__ = ["BoTorchMFKG"]


class BoTorchMFKG(Generator):
    """
    Multi-fidelity Bayesian optimization using BoTorch's MFKG acquisition function.

    The VOCS must contain:
    - At least two design variables (any names).
    - A variable named ``fidelity`` with domain ``[0, 1]``.
    - Exactly one objective.

    On the first call to ``suggest``, ``2 * n_init_samples`` points are returned:
    each of the ``n_init_samples`` random design points is evaluated at both low
    (fidelity=0) and high (fidelity=1) fidelity.  After that, each ``suggest``
    call fits the GP, builds the MFKG acquisition function, and returns ``q``
    candidates chosen by ``optimize_acqf_mixed``.

    Because the internal optimiser always proposes exactly ``q`` candidates,
    ``suggest(n_trials)`` returns ``min(n_trials, q)`` of them.  Set
    ``GenSpecs.batch_size = q`` so libEnsemble always requests the right number.

    Args:
        vocs: VOCS object defining variables (must include ``fidelity``), objectives.
        n_init_samples: Number of random design points for the initial batch.
            Each is evaluated at both fidelities, so ``2 * n_init_samples``
            simulations are submitted on the first ``suggest`` call.
        q: Number of MFKG candidates to propose per subsequent ``suggest`` call.
        fidelity_weights: Mapping of fidelity-dimension index to weight, passed to
            ``AffineFidelityCostModel``.  Defaults to ``{fidelity_dim: 0.9}``.
        fixed_cost: Fixed cost term for ``AffineFidelityCostModel``.
        num_fantasies: Number of fantasy samples for MFKG.
        num_restarts: ``num_restarts`` for BoTorch optimisation routines.
        raw_samples: ``raw_samples`` for BoTorch optimisation routines.
        seed: Random seed for reproducibility.
        fidelity_variable: Name of the fidelity variable in VOCS.  Defaults to
            ``"fidelity"``.
    """

    def __init__(
        self,
        vocs: VOCS,
        n_init_samples: int = 4,
        q: int = 2,
        fidelity_weights: dict | None = None,
        fixed_cost: float = 0.1,
        num_fantasies: int = 128,
        num_restarts: int = 1,
        raw_samples: int = 10,
        seed: int = 42,
        fidelity_variable: str = "fidelity",
        *args,
        **kwargs,
    ):
        self.vocs = vocs
        self.n_init_samples = n_init_samples
        self.q = q
        self.num_fantasies = num_fantasies
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.fidelity_variable = fidelity_variable
        self._initialized = False

        # Torch device / dtype settings
        self._tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        torch.manual_seed(seed)

        # Derive variable ordering from VOCS (preserves insertion order)
        self._var_names = list(vocs.variables.keys())
        self._fidelity_dim = self._var_names.index(fidelity_variable)
        self._design_var_names = [v for v in self._var_names if v != fidelity_variable]
        self._n_design = len(self._design_var_names)

        # Build bounds tensor: shape (2, n_vars)
        lowers = [vocs.variables[v].domain[0] for v in self._var_names]
        uppers = [vocs.variables[v].domain[1] for v in self._var_names]
        self._bounds = torch.tensor([lowers, uppers], **self._tkwargs)

        # Target fidelity maps fidelity_dim → 1.0
        self._target_fidelities = {self._fidelity_dim: 1.0}

        # Cost model and cost-aware utility
        if fidelity_weights is None:
            fidelity_weights = {self._fidelity_dim: 0.9}
        self._cost_model = AffineFidelityCostModel(fidelity_weights=fidelity_weights, fixed_cost=fixed_cost)
        self._cost_aware_utility = InverseCostWeightedUtility(self._cost_model)

        # Accumulated training data (populated by ingest)
        self._train_x: torch.Tensor | None = None
        self._train_obj: torch.Tensor | None = None

        # Pending candidates proposed in the most recent suggest call, waiting
        # for ingest to receive their objective values.
        self._pending_x: torch.Tensor | None = None

        super().__init__(vocs, *args, **kwargs)

    # ------------------------------------------------------------------
    # gest-api interface
    # ------------------------------------------------------------------

    def _validate_vocs(self, vocs: VOCS) -> None:
        assert len(vocs.variable_names) >= 2, "VOCS must have at least two variables."
        assert (
            self.fidelity_variable in vocs.variables
        ), f"VOCS must contain a variable named '{self.fidelity_variable}'."
        assert len(vocs.objective_names) == 1, "VOCS must contain exactly one objective."

    def suggest(self, n_trials: int) -> list[dict]:
        """
        Return up to ``n_trials`` candidate points.

        On the first call, returns ``2 * n_init_samples`` initial points
        (random design coordinates evaluated at both fidelities).  On
        subsequent calls, fits the MFKG model and returns ``q`` candidates,
        capped at ``n_trials``.
        """
        if not self._initialized:
            candidates = self._initial_candidates()
            self._initialized = True
        else:
            candidates = self._mfkg_candidates()

        # Respect n_trials: never return more than requested
        candidates = candidates[:n_trials]
        self._pending_x = candidates
        return self._tensor_to_dicts(candidates)

    def ingest(self, results: list[dict]) -> None:
        """
        Receive evaluated objective values and append to training data.

        Args:
            results: List of dicts, each containing variable names and the
                objective name(s) as keys.
        """
        if not results:
            return

        obj_name = self.vocs.objective_names[0]

        # Reconstruct x tensor in the same variable order used for suggest
        x_rows = []
        obj_rows = []
        for r in results:
            x_row = [r[v] for v in self._var_names]
            x_rows.append(x_row)
            obj_rows.append([r[obj_name]])

        new_x = torch.tensor(x_rows, **self._tkwargs)
        new_obj = torch.tensor(obj_rows, **self._tkwargs)

        if self._train_x is None:
            self._train_x = new_x
            self._train_obj = new_obj
        else:
            self._train_x = torch.cat([self._train_x, new_x])
            self._train_obj = torch.cat([self._train_obj, new_obj])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_candidates(self) -> torch.Tensor:
        """
        Generate 2 * n_init_samples initial candidates: each of n_init_samples
        random design points is duplicated at low (0) and high (1) fidelity.
        """
        design_lowers = [self.vocs.variables[v].domain[0] for v in self._design_var_names]
        design_uppers = [self.vocs.variables[v].domain[1] for v in self._design_var_names]
        lb = torch.tensor(design_lowers, **self._tkwargs)
        ub = torch.tensor(design_uppers, **self._tkwargs)

        design_pts = lb + (ub - lb) * torch.rand(self.n_init_samples, self._n_design, **self._tkwargs)

        lf = torch.zeros(self.n_init_samples, 1, **self._tkwargs)
        hf = torch.ones(self.n_init_samples, 1, **self._tkwargs)

        # Insert fidelity column at the correct position
        lf_pts = self._insert_fidelity_col(design_pts, lf)
        hf_pts = self._insert_fidelity_col(design_pts, hf)

        return torch.cat([lf_pts, hf_pts], dim=0)

    def _mfkg_candidates(self) -> torch.Tensor:
        """Fit the GP, build MFKG, and optimise to get q candidates."""
        mll, model = self._initialize_model(self._train_x, self._train_obj)
        fit_gpytorch_mll(mll)
        mfkg_acqf = self._get_mfkg(model)
        candidates = self._optimize_mfkg(mfkg_acqf)
        return candidates.detach()

    def _insert_fidelity_col(self, design: torch.Tensor, fidelity: torch.Tensor) -> torch.Tensor:
        """Reassemble full variable tensor inserting fidelity at the correct column."""
        n = design.shape[0]
        full = torch.empty(n, len(self._var_names), **self._tkwargs)
        design_col = 0
        for col, name in enumerate(self._var_names):
            if name == self.fidelity_variable:
                full[:, col] = fidelity.squeeze(-1)
            else:
                full[:, col] = design[:, design_col]
                design_col += 1
        return full

    def _initialize_model(self, train_x: torch.Tensor, train_obj: torch.Tensor) -> tuple:
        model = SingleTaskMultiFidelityGP(
            train_x,
            train_obj,
            outcome_transform=Standardize(m=1),
            data_fidelities=[self._fidelity_dim],
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def _get_mfkg(self, model) -> qMultiFidelityKnowledgeGradient:
        """Build the MFKG acquisition function."""
        # Estimate current value at target fidelity
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=len(self._var_names),
            columns=[self._fidelity_dim],
            values=[1],
        )
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self._bounds[:, [i for i in range(len(self._var_names)) if i != self._fidelity_dim]],
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 10, "maxiter": 10},
        )

        n_vars = len(self._var_names)

        def _project(X):
            return project_to_target_fidelity(X=X, target_fidelities=self._target_fidelities, d=n_vars)

        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=self.num_fantasies,
            current_value=current_value,
            cost_aware_utility=self._cost_aware_utility,
            project=_project,
        )

    def _optimize_mfkg(self, mfkg_acqf) -> torch.Tensor:
        """Run optimize_acqf_mixed to propose q candidates."""
        candidates, _ = optimize_acqf_mixed(
            acq_function=mfkg_acqf,
            bounds=self._bounds,
            fixed_features_list=[{self._fidelity_dim: 0.0}, {self._fidelity_dim: 1.0}],
            q=self.q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 10, "maxiter": 10},
        )
        return candidates

    def _tensor_to_dicts(self, candidates: torch.Tensor) -> list[dict]:
        """Convert a (n, n_vars) tensor to a list of variable-name dicts."""
        result = []
        candidates_np = candidates.cpu().numpy()
        for row in candidates_np:
            d = {name: float(row[i]) for i, name in enumerate(self._var_names)}
            result.append(d)
        return result
