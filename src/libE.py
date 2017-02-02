"""
Main libEnsemble routine 
============================================

First tasks to be included in libEnsemble: [depends on]
    - generate input for simulations 
    - launch simulations
    - receive ouptut from simulations [sim]
    - coordinate concurrent simulations [sim]
    - track simulation history

Second tasks to be included in libEnsemble:
    - allocate resources to simulations [machinefiles or sim or queue system]
    - receive intermediate output from simulations [sim]
    - tolerate failed simulations [MPI/Hydra]
    - exploit persistent data (e.g., checkpointing, meshes, iterative solvers) [sim & OS/memory]
    - gracefully terminate simulations [sim & ??]

Third (aspirational) tasks for libEnsemble:
    - change resources given to a simulation midrun [sim]
    - recover resources when simulation fails [MPI/Hydra]
    - gracefully pause cimulation [sim]

"""


def libE(comm, history, allocation_specs, sim_specs, failure_processing,
        exit_criteria):

    """ 
    Parameters
    ----------

    comm: [mpi4py communicator] to be used by libE


    history: [numpy structured array] with fields matching those returned from libE
        - x: parameters given to simulations
        - f: simulation value(s) at each x
        - ...
        - ...

    allocation_specs: [dict]
        - ranks
        - machinefile

    sim_specs: [dict of dicts] one dict for each simulation.
        Possible fields of a simulation's dict:
        - sim_f: [func] that calls sim
        - sim_f_params: [dict] parameters for sim_f
            - n: [int] dimension of simulation parameters
            - m: [int] dimension of simulation output 
            - lb: [n-by-1 array] lower bound on sim parameters
            - ub: [n-by-1 array] upper bound on sim parameters
            - data: 
        - gen_f: [func] generates next points to be evaluated by a sim
        - gen_f_params: [dict] parameters for gen_f

        Possible fields of a local optimization's dict:
        - various tolerances and settings 
       
    failure_processing: [dict]
        - 

    exit_criteria: [dict] with possible fields:
        - sim_eval_max: [int] 




    Possible sim_f API:


    sim_f(input_params, resources, progress_handles):
        # Code 
        # to 
        # do 
        # simulation
        return(output_params)
    """
    

