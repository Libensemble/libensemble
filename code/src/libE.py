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

from mpi4py import MPI

def libE(sim_specs, gen_specs, exit_criteria, failure_processing={},
        allocation_specs={'manager_ranks': set([0]), 'worker_ranks': set(range(1,MPI.COMM_WORLD.Get_size()))},
        c={'comm': MPI.COMM_WORLD, 'color': 0}, 
        H0=[]):

    """ 
    Parameters
    ----------
    """
    check_inputs(c, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria)
    
    comm = c['comm']
    comm.Barrier()

    if comm.Get_rank() in allocation_specs['manager_ranks']:
        H = manager_main(comm, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria, H0)
    elif comm.Get_rank() in allocation_specs['worker_ranks']:
        worker_main(c)
        H = []
    else:
        print("Rank: %d not manager or worker" % comm.Get_rank())

    comm.Barrier()
    return(H)


def check_inputs(c, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria):
    valid_term_fields = ['sim_max','gen_max','elapsed_wallclock_time','stop_val']
    assert(any([term_field in exit_criteria for term_field in valid_term_fields])), "Must have a valid termination option: " + valid_term_fields 

    assert(len(sim_specs['out'])), "sim_specs must have 'out' entries"
    assert(len(gen_specs['out'])), "gen_specs must have 'out' entries"
    assert(len(allocation_specs['worker_ranks'])), "Must have at least one worker rank"
    assert(len(allocation_specs['manager_ranks'])), "Must have at least one manager rank"

    if 'stop_val' in exit_criteria:
        assert(exit_criteria['stop_val'][0] in [e[0] for e in sim_specs['out']] + [e[0] for e in gen_specs['out']]),\
               "Can't stop on " + exit_criteria['stop_val'][0] + " if it's not \
               returned from sim_specs['out'] or gen_specs['out']"
    
    if 'num_inst' in gen_specs and 'batch_mode' in gen_specs:
        assert(gen_specs['num_inst'] <= 1 or not gen_specs['batch_mode']),\
               "Can't have more than one 'num_inst' for 'batch_mode' generator"

    if 'single_component_at_a_time' in gen_specs['params'] and gen_specs['params']['single_component_at_a_time']:
        if gen_specs['gen_f'].__name__ == 'aposmm_logic':
            assert(gen_specs['batch_mode']), "Must be in batch mode when using 'single_component_at_a_time' and APOSMM"

