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

from __future__ import division
from __future__ import absolute_import

from libE_manager import manager_main
from libE_worker import worker_main

def libE(c, history, allocation_specs, sim_specs, failure_processing,
        exit_criteria):

    """ 
    Parameters
    ----------

    c['comm']: [mpi4py communicator] to be used by libE
    c['color']: [int] communicator color


    history: [numpy structured array] with fields matching those returned from libE
        - x: parameters given to simulations
        - f: simulation value(s) at each x
        - ...
        - ...

    allocation_specs: [dict]
        - manager_ranks: [python set of ints] 
        - lead_worker_ranks: [python set of ints]
        - machinefile:

    sim_specs: [dict of dicts] one dict for each simulation.
        Possible fields of a simulation's dict:
        - sim_f: [func] that calls sim
        - sim_f_params: [dict] parameters for sim_f
            - n: [int] dimension of simulation parameters
            - m: [int] dimension of simulation output 
            - data: 
        - gen_f: [func] generates next points to be evaluated by a sim
        - gen_f_params: [dict] parameters for gen_f
            - lb: [n-by-1 array] lower bound on sim parameters
            - ub: [n-by-1 array] upper bound on sim parameters

        Possible fields of a local optimization's dict:
        - various tolerances and settings 
       
    failure_processing: [dict]
        - 

    exit_criteria: [dict] with possible fields:
        - sim_eval_max: [int] Stop after this many evaluations.
        - min_sim_f_val: [dbl] Stop when a value below this has been found.



    Possible sim_f API:


    sim_f(input_params, resources, progress_handles):
        # Code 
        # to 
        # do 
        # simulation
        return(output_params)
    
    """
    comm = c['comm']

    comm.Barrier()

    if comm.Get_rank() in allocation_specs['manager_ranks']:
        H = manager_main(comm, history, allocation_specs, sim_specs, failure_processing, exit_criteria)
        print(min(H['f']))
        print(len(H['f']))
    elif comm.Get_rank() in allocation_specs['lead_worker_ranks']:
        worker_main(c, allocation_specs, sim_specs, failure_processing)
    else:
        print("Rank: %d not manager, custodian, or worker" % comm.Get_rank())

    comm.Barrier()

    
