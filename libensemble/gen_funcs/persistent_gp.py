"""
This file defines the `gen_f` for Bayesian optimization with a Gaussian process.

The `gen_f` is called once by a dedicated worker and only returns at the end
of the whole libEnsemble run.

This `gen_f` is meant to be used with the `alloc_f` function
`only_persistent_gens`
"""

from argparse import Namespace

import numpy as np
from dragonfly.exd.cp_domain_utils import load_config

# import dragonfly Gaussian Process functions
from dragonfly.exd.domains import EuclideanDomain
from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from dragonfly.opt.gp_bandit import CPGPBandit, EuclideanGPBandit

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport


def initialize(U, libE_info):
    # Extract bounds of the parameter space, and batch size
    ub_list = U.get("ub", None)
    lb_list = U.get("lb", None)
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Fidelity range.
    fidel_range = U.get("range", None)

    # Get fidelity cost function.
    cost_func = U.get("cost_func", None)

    # Number of points to generate initially
    number_of_gen_points = U.get("gen_batch_size", None)

    return ub_list, lb_list, ps, fidel_range, cost_func, number_of_gen_points


def feed_GP(H, opt, with_z=True):
    for i in range(len(H)):
        x = H["x"][i]
        y = H["f"][i]
        if with_z:
            z = H["z"][i]
            opt.tell([([z], x, -y)])
        else:
            opt.tell([(x, -y)])
    # Update hyperparameters
    opt._build_new_model()
    return opt


def build_requested_points(number_of_gen_points, out_dtype, opt, set_type=1):
    H_o = np.zeros(number_of_gen_points, dtype=out_dtype)
    for i in range(number_of_gen_points):
        if set_type == 0:
            x = opt.ask()
            H_o["x"][i] = x
            H_o["resource_sets"][i] = 1
        elif set_type == 1:
            z, input_vector = opt.ask()
            H_o["x"][i] = input_vector
            H_o["z"][i] = z[0]
            H_o["resource_sets"][i] = max(1, int(z[0] / 2))

    return H_o


def persistent_gp_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, update it as new simulation results
    are available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    ub_list, lb_list, ps, _, _, number_of_gen_points = initialize(gen_specs["user"], libE_info)

    # Initialize the dragonfly GP optimizer
    domain = EuclideanDomain([[lo, up] for lo, up in zip(lb_list, ub_list)])
    func_caller = EuclideanFunctionCaller(None, domain)
    opt = EuclideanGPBandit(
        func_caller,
        ask_tell_mode=True,
        options=Namespace(
            acq="ts",
            build_new_model_every=number_of_gen_points,
            init_capital=number_of_gen_points,
        ),
    )
    opt.initialise()

    # If there is any past history, feed it to the GP
    if len(H) > 0:
        opt = feed_GP(H, opt, with_z=False)

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = build_requested_points(number_of_gen_points, gen_specs["out"], opt, set_type=0)

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in["f"])
            # Update the GP with latest simulation results
            opt = feed_GP(calc_in, opt, with_z=False)
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_gp_mf_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, for multi-fidelity optimization,
    and update it as new simulation results are available, and generate
    inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """

    ub_list, lb_list, ps, fidel_range, cost_func, number_of_gen_points = initialize(gen_specs["user"], libE_info)

    # Initialize the dragonfly GP optimizer
    domain = EuclideanDomain([[lo, up] for lo, up in zip(lb_list, ub_list)])
    fidel_space = EuclideanDomain([fidel_range])
    func_caller = EuclideanFunctionCaller(
        None,
        raw_domain=domain,
        raw_fidel_space=fidel_space,
        fidel_cost_func=cost_func,
        raw_fidel_to_opt=fidel_range[-1],
    )
    opt = EuclideanGPBandit(
        func_caller,
        ask_tell_mode=True,
        is_mf=True,
        options=Namespace(
            acq="ts",
            build_new_model_every=number_of_gen_points,
            init_capital=number_of_gen_points,
        ),
    )
    opt.initialise()

    # If there is any past history, feed it to the GP
    if len(H) > 0:
        opt = feed_GP(H, opt)

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = build_requested_points(number_of_gen_points, gen_specs["out"], opt, set_type=1)

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in["f"])
            # Update the GP with latest simulation results
            opt = feed_GP(calc_in, opt)
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_gp_mf_disc_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, for multi-fidelity optimization
    in a discrete fidelity space, and update it as new simulation results are
    available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    ub_list, lb_list, ps, fidel_range, cost_func, number_of_gen_points = initialize(gen_specs["user"], libE_info)

    # Create configuration dictionary from which Dragongly will
    # automatically generate the necessary domains and orderings.
    config_params = {}
    config_params["domain"] = []
    for ub, lb in zip(ub_list, lb_list):
        domain_dict = {
            "max": ub,
            "min": lb,
            "type": "float",
        }
        config_params["domain"].append(domain_dict)
    config_params["fidel_space"] = [
        {
            "type": "discrete",
            "items": fidel_range,
        }
    ]
    config_params["fidel_to_opt"] = [fidel_range[-1]]
    config = load_config(config_params)

    # Initialize the dragonfly GP optimizer.
    func_caller = CPFunctionCaller(
        None,
        domain=config.domain,
        domain_orderings=config.domain_orderings,
        fidel_space=config.fidel_space,
        fidel_cost_func=cost_func,
        fidel_to_opt=config.fidel_to_opt,
        fidel_space_orderings=config.fidel_space_orderings,
    )

    opt = CPGPBandit(
        func_caller,
        ask_tell_mode=True,
        is_mf=True,
        options=Namespace(
            acq="ts",
            build_new_model_every=number_of_gen_points,
            init_capital=number_of_gen_points,
        ),
    )
    opt.initialise()

    # If there is any past history, feed it to the GP
    if len(H) > 0:
        opt = feed_GP(H, opt)

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = build_requested_points(number_of_gen_points, gen_specs["out"], opt, set_type=1)

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in["f"])
            # Update the GP with latest simulation results
            opt = feed_GP(calc_in, opt)
            # Update hyperparameters
            opt._build_new_model()
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
