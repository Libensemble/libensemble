"""
This file defines the `gen_f` for Bayesian optimization with a Gaussian process
and the multi-task algorithm of Ax.

The `gen_f` is called once by a dedicated worker and only returns at the end
of the whole libEnsemble run.

This `gen_f` is meant to be used with the `alloc_f` function
`only_persistent_gens`

Ax notes:
Each arm = a set of simulation inputs (a sim_id)
Each trial = a batch of simulations.
The metric = the recorded simulation output (f) that Ax optimizes.
Ax runner handles the execution of trials - AxRunner wraps Runner to use libE tr.run()

"""

import os
from copy import deepcopy
from typing import Optional
from pyre_extensions import assert_is_instance
import warnings

import numpy as np
import pandas as pd
import torch

from ax import Metric, Runner
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import Objective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import AxParameterWarning
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.registry import Models, ST_MTGP_trans
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.convert_metric_names import tconfig_from_mt_experiment
from ax.storage.metric_registry import register_metrics
from ax.runners import SyntheticRunner
from ax.storage.json_store.save import save_experiment
from ax.storage.runner_registry import register_runner
from ax.utils.common.result import Ok

try:
    # For Ax >= 0.5.0
    from ax.modelbridge.transforms.derelativize import Derelativize
    from ax.modelbridge.transforms.convert_metric_names import ConvertMetricNames
    from ax.modelbridge.transforms.trial_as_task import TrialAsTask
    from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
    from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice
    from ax.modelbridge.registry import MBM_X_trans
    MT_MTGP_trans = MBM_X_trans + [
        Derelativize,
        ConvertMetricNames,
        TrialAsTask,
        StratifiedStandardizeY,
        TaskChoiceToIntTaskChoice,
    ]

except ImportError:
    # For Ax < 0.5.0
    from ax.modelbridge.registry import MT_MTGP_trans

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport

__all__ = ["persistent_gp_mt_ax_gen_f"]

warnings.filterwarnings(
    "ignore",
    message="`cache_root` is only supported for GPyTorchModels",
    category=RuntimeWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Changing `is_ordered` to `True` for `ChoiceParameter`",
    category=AxParameterWarning,
)


# get_MTGP based on https://ax.dev/docs/tutorials/multi_task/
def get_MTGP(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    trial_index: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.double,
) -> TorchModelBridge:
    """Instantiates a Multi-task Gaussian Process (MTGP) model that generates
    points with EI.

    If the input experiment is a MultiTypeExperiment then a
    Multi-type Multi-task GP model will be instantiated.
    Otherwise, the model will be a Single-type Multi-task GP.
    """

    if isinstance(experiment, MultiTypeExperiment):
        trial_index_to_type = {
            t.index: t.trial_type for t in experiment.trials.values()
        }
        transforms = MT_MTGP_trans
        transform_configs = {
            "TrialAsTask": {"trial_level_map": {"trial_type": trial_index_to_type}},
            "ConvertMetricNames": tconfig_from_mt_experiment(experiment),
        }
    else:
        # Set transforms for a Single-type MTGP model.
        transforms = ST_MTGP_trans
        transform_configs = None

    # Choose the status quo features for the experiment from the selected trial.
    # If trial_index is None, we will look for a status quo from the last
    # experiment trial to use as a status quo for the experiment.
    if trial_index is None:
        trial_index = len(experiment.trials) - 1
    elif trial_index >= len(experiment.trials):
        raise ValueError("trial_index is bigger than the number of experiment trials")

    status_quo = experiment.trials[trial_index].status_quo
    if status_quo is None:
        status_quo_features = None
    else:
        status_quo_features = ObservationFeatures(
            parameters=status_quo.parameters,
            trial_index=trial_index,  # pyre-ignore[6]
        )

    return assert_is_instance(
        Models.ST_MTGP(
            experiment=experiment,
            search_space=search_space or experiment.search_space,
            data=data,
            transforms=transforms,
            transform_configs=transform_configs,
            torch_dtype=dtype,
            torch_device=device,
            status_quo_features=status_quo_features,
        ),
        TorchModelBridge,
    )


def persistent_gp_mt_ax_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model for multi-task optimization
    and update it as new simulation results are
    available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs["user"]["ub"]
    lb_list = gen_specs["user"]["lb"]

    # Get task names.
    hifi_task = gen_specs["user"]["name_hifi"]
    lofi_task = gen_specs["user"]["name_lofi"]

    # Number of points to generate initially and during optimization.
    n_init_hifi = gen_specs["user"]["n_init_hifi"]
    n_init_lofi = gen_specs["user"]["n_init_lofi"]
    n_opt_hifi = gen_specs["user"]["n_opt_hifi"]
    n_opt_lofi = gen_specs["user"]["n_opt_lofi"]

    # Create search space.
    parameters = []
    for i, (ub, lb) in enumerate(zip(ub_list, lb_list)):
        parameters.append(
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=float(lb),
                upper=float(ub),
            )
        )
    search_space = SearchSpace(parameters=parameters)

    # Create metrics.
    hifi_objective = AxMetric(name="hifi_metric", lower_is_better=True)
    lofi_objective = AxMetric(name="lofi_metric", lower_is_better=True)

    # Create optimization config.
    opt_config = OptimizationConfig(objective=Objective(hifi_objective, minimize=True))

    # Create runner.
    ax_runner = AxRunner(libE_info, gen_specs)

    # Create experiment.
    exp = MultiTypeExperiment(
        name="mt_exp",
        search_space=search_space,
        default_trial_type=hifi_task,
        default_runner=ax_runner,
        optimization_config=opt_config,
    )

    # hifi_task has been added as default but we need to add lofi task and link them.
    exp.add_trial_type(lofi_task, ax_runner)
    exp.add_tracking_metric(metric=lofi_objective, trial_type=lofi_task, canonical_name="hifi_metric")

    # TODO: Implement reading past history (by reading saved experiment or
    # libEnsemble history file).

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = 0
    hifi_trials = []
    while tag not in [STOP_TAG, PERSIS_STOP]:
        if model_iteration == 0:
            # Initialize with sobol sample.
            for model, n_gen in zip([hifi_task, lofi_task], [n_init_hifi, n_init_lofi]):
                s = get_sobol(exp.search_space, scramble=False)
                gr = s.gen(n_gen)
                trial = exp.new_batch_trial(trial_type=model, generator_run=gr)
                trial.run()
                trial.mark_completed()
                tag = trial.run_metadata["tag"]
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break
                if model == hifi_task:
                    hifi_trials.append(trial.index)

        else:
            # Run multi-task BO.

            # Fit the MTGP.
            m = get_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                search_space=exp.search_space,
            )

            # Find the best points for the high fidelity task.
            gr = m.gen(
                n=n_opt_lofi,
                optimization_config=exp.optimization_config,
                fixed_features=ObservationFeatures(parameters={}, trial_index=hifi_trials[-1]),
            )

            # But launch them at low fidelity.
            tr = exp.new_batch_trial(trial_type=lofi_task, generator_run=gr)
            tr.run()  # Runs sims via libE (see AxRunner.run below)
            tr.mark_completed()
            tag = tr.run_metadata["tag"]
            if tag in [STOP_TAG, PERSIS_STOP]:
                break

            # Update the model.
            m = get_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                search_space=exp.search_space,
            )

            # Select max-utility points from the low fidelity batch to generate a high fidelity batch.
            gr = max_utility_from_GP(n=n_opt_hifi, m=m, gr=gr, hifi_task=hifi_task)
            tr = exp.new_batch_trial(trial_type=hifi_task, generator_run=gr)
            tr.run()  # Runs sims via libE (see AxRunner.run below)
            tr.mark_completed()
            tag = tr.run_metadata["tag"]
            if tag in [STOP_TAG, PERSIS_STOP]:
                break
            hifi_trials.append(tr.index)

        if model_iteration == 0:
            # Initialize folder to log the model.
            if not os.path.exists("model_history"):
                os.mkdir("model_history")
            # Register metric and runner in order to be able to save to json.
            _, encoder_registry, decoder_registry = register_metrics(
                {AxMetric: None}
            )
            _, encoder_registry, decoder_registry = register_runner(
                AxRunner,
                encoder_registry=encoder_registry,
                decoder_registry=decoder_registry,
            )

        # Save current experiment.
        # Saving the experiment to a json file currently requires a bit of
        # trickery. The `AxRunner` cannot be serialized into a json file
        # due to the `libE_info` and `gen_specs` attributes. This also prevents
        # the experiment from being saved to file. In order to overcome this,
        # all instances of the `AxRunner` are replaced by a `SyntheticRunner`
        # before saving. Afterwards, the `AxRunner` is reasigned again to both
        # high- and low-fidelity tasks in order to allow the optimization to
        # continue.
        for i, trial in exp.trials.items():
            trial._runner = SyntheticRunner()
        exp.update_runner(lofi_task, SyntheticRunner())
        exp.update_runner(hifi_task, SyntheticRunner())
        save_experiment(exp, "model_history/experiment_%05d.json" % model_iteration, encoder_registry=encoder_registry)
        exp.update_runner(lofi_task, ax_runner)
        exp.update_runner(hifi_task, ax_runner)

        # Increase iteration counter.
        model_iteration += 1

    return [], persis_info, FINISHED_PERSISTENT_GEN_TAG


class AxRunner(Runner):
    """Custom runner in charge of executing the trials using libEnsemble."""

    def __init__(self, libE_info, gen_specs):
        self.libE_info = libE_info
        self.gen_specs = gen_specs
        self.ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
        super().__init__()

    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        task = trial.trial_type
        number_of_gen_points = len(trial.arms)
        H_o = np.zeros(number_of_gen_points, dtype=self.gen_specs["out"])

        for i, (arm_name, arm) in enumerate(trial.arms_by_name.items()):
            # fill H_o
            params = arm.parameters
            n_param = len(params)
            param_array = np.zeros(n_param)
            for j in range(n_param):
                param_array[j] = params[f"x{j}"]
            H_o["x"][i] = param_array
            H_o["resource_sets"][i] = 1  # one is default but could be diff for hi/lo
            H_o["task"][i] = task
        tag, Work, calc_in = self.ps.send_recv(H_o)

        trial_metadata["tag"] = tag
        for i, (arm_name, arm) in enumerate(trial.arms_by_name.items()):
            # fill metadata
            params = arm.parameters
            trial_metadata[arm_name] = {
                "arm_name": arm_name,
                "trial_index": trial.index,
                "f": calc_in["f"][i] if calc_in is not None else None,
            }
        return trial_metadata


class AxMetric(Metric):
    """Custom metric to be optimized during the experiment."""

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": trial.run_metadata[arm_name]["f"],
                    "sem": 0.0,
                }
            )
        data = Data(df=pd.DataFrame.from_records(records))
        return Ok(data)


def max_utility_from_GP(n, m, gr, hifi_task):
    """
    High fidelity batches are constructed by selecting the maximum utility points
    from the low fidelity batch, after updating the model with the low fidelity results.
    This function selects the max utility points according to the MTGP
    predictions.
    """
    obsf = []
    for arm in gr.arms:
        params = deepcopy(arm.parameters)
        params["trial_type"] = hifi_task
        obsf.append(ObservationFeatures(parameters=params))
    # Make predictions
    f, cov = m.predict(obsf)
    # Compute expected utility
    u = -np.array(f["hifi_metric"])
    best_arm_indx = np.flip(np.argsort(u))[:n]
    gr_new = GeneratorRun(
        arms=[gr.arms[i] for i in best_arm_indx],
        weights=[1.0] * n,
    )
    return gr_new
