"""
This file defines the `gen_f` for Bayesian optimization with a Gaussian process.

The `gen_f` is called once by a dedicated worker and only returns at the end
of the whole libEnsemble run.

This `gen_f` is meant to be used with the `alloc_f` function
`only_persistent_gens`
"""

import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

# import dragonfly Gaussian Process functions
from dragonfly.exd.domains import EuclideanDomain
from dragonfly.exd.experiment_caller import EuclideanFunctionCaller
from dragonfly.opt.gp_bandit import EuclideanGPBandit
from argparse import Namespace

def persistent_gp_gen_f( H, persis_info, gen_specs, libE_info ):
    """
    Create a Gaussian Process model, update it as new simulation results
    are available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs['user']['ub']
    lb_list = gen_specs['user']['lb']

    # Number of points to generate initially
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    # Initialize the dragonfly GP optimizer
    domain = EuclideanDomain( [ [l,u] for l,u in zip(lb_list, ub_list) ] )
    func_caller = EuclideanFunctionCaller(None, domain)
    opt = EuclideanGPBandit( func_caller, ask_tell_mode=True,
          options=Namespace(acq='ts', build_new_model_every=number_of_gen_points) )
    opt.initialise()

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            x = opt.ask()
            H_o['x'][i] = x

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['f'])
            # Update the GP with latest simulation results
            for i in range(n):
                x = calc_in['x'][i]
                y = calc_in['f'][i]
                opt.tell([ (x, -y) ])
            # Update hyperparameters
            opt._build_new_model()
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG



def persistent_gp_mf_gen_f( H, persis_info, gen_specs, libE_info ):
    """
    Create a Gaussian Process model, for multi-fidelity optimization,
    and update it as new simulation results are available, and generate
    inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs['user']['ub']
    lb_list = gen_specs['user']['lb']
    # Note: the resolution should be the last variable
    # TODO: Add a corresponding automated check

    # Hard-coded cost function: TODO: generalize, allow user to pass
    cost_func = lambda z: z[0]**2

    # Number of points to generate initially
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    # Initialize the dragonfly GP optimizer
    domain = EuclideanDomain( [ [l,u] for l,u in zip(lb_list, ub_list)[:-1] ] )
    fidel_space = EuclideanDomain( [ [lb_list[-1], ub_list[-1]] ] )
    func_caller = EuclideanFunctionCaller( None,
                            raw_domain=domain,
                            raw_fidel_space=fidel_space,
                            fidel_cost_func=cost_func,
                            raw_fidel_to_opt=ub_list[-1] )
    opt = EuclideanGPBandit( func_caller,
                            ask_tell_mode=True,
                            is_mf=True,
                            options=Namespace(acq='ts',
                            build_new_model_every=number_of_gen_points) )
    opt.initialise()

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            resolution, input_vector = opt.ask()
            H_o['x'][i] = np.concatenate( input_vector, resolution )

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['f'])
            # Update the GP with latest simulation results
            for i in range(n):
                x = calc_in['x'][i]
                input_vector = x[:-1]
                resolution = x[-1]
                y = calc_in['f'][i]
                opt.tell([ (resolution, input_vector, -y) ])
            # Update hyperparameters
            opt._build_new_model()
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
