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

def libE(c, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria):

    """ 
    Parameters
    ----------

    """
    check_inputs(c, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria)
    
    comm = c['comm']
    comm.Barrier()

    if comm.Get_rank() in allocation_specs['manager_ranks']:
        H = manager_main(comm, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria)
        # print(H)
        # print(H.dtype.names)
    elif comm.Get_rank() in allocation_specs['worker_ranks']:
        worker_main(c)
        H = []
    else:
        print("Rank: %d not manager, custodian, or worker" % comm.Get_rank())

    comm.Barrier()
    return(H)


def check_inputs(c, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria):

    assert(len(sim_specs['out'])), "sim_specs must have 'out' entries"
    assert(len(gen_specs['out'])), "gen_specs must have 'out' entries"

    if 'stop_val' in exit_criteria:
        assert(exit_criteria['stop_val'](0) in sim_specs['out'] + gen_specs['out']),\
               "Can't stop on " + exit_criteria['stop_val'][0] + " if it's not \
               returned from sim_specs['out'] or gen_specs['out']"
    
    if 'num_inst' in gen_specs and 'batch_mode' in gen_specs:
        assert(gen_specs['num_inst'] <= 1 or not gen_specs['batch_mode']),\
               "Can't have more than one 'num_inst' for 'batch_mode' generator"


