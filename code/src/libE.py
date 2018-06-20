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

import numpy as np
import sys,os 

# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

sys.path.append(os.path.join(os.path.dirname(__file__), '../examples/alloc_funcs'))
from give_sim_work_first import give_sim_work_first

def libE(sim_specs, gen_specs, exit_criteria, failure_processing={},
        alloc_specs={'alloc_f': give_sim_work_first, 'out':[]} ,
        libE_specs={'comm': MPI.COMM_WORLD, 'color': 0, 'manager_ranks': set([0]), 'worker_ranks': set(range(1,MPI.COMM_WORLD.Get_size()))}, 
        H0=[]):
    """ 
    This is the outer libEnsemble routine. It checks each rank in libE_specs['comm']
    against libE_specs['manager_ranks'] or libE_specs['worker_ranks'] and
    either runs manager_main or worker_main 
    (Some subroutines currently assume that the manager is always (only) rank 0.)
    """

    libE_specs = check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, failure_processing, exit_criteria, H0)
    
    # When timing libEnsemble, uncomment barrier to ensure manager and workers are in sync
    # libE_specs['comm'].Barrier()

    if libE_specs['comm'].Get_rank() in libE_specs['manager_ranks']:
        H, gen_info, exit_flag = manager_main(libE_specs, alloc_specs, sim_specs, gen_specs, failure_processing, exit_criteria, H0)
        # if exit_flag == 0:
        #     libE_specs['comm'].Barrier()
    elif libE_specs['comm'].Get_rank() in libE_specs['worker_ranks']:
        worker_main(libE_specs, sim_specs, gen_specs); H=gen_info=exit_flag=[]
        # libE_specs['comm'].Barrier()
    else:
        print("Rank: %d not manager or worker" % libE_specs['comm'].Get_rank()); H=gen_info=exit_flag=[]

    return H, gen_info, exit_flag




def check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, failure_processing, exit_criteria, H0):
    """ 
    Check if the libEnsemble arguments are of the correct data type contain
    sufficient information to perform a run. 
    """

    if 'comm' not in libE_specs and ('manager_ranks' in libE_specs or 'worker_ranks' in libE_specs):
        sys.exit('Must give a communicator when specifying manager and worker ranks')

    if 'comm' not in libE_specs:
        libE_specs['comm'] = MPI.COMM_WORLD
        libE_specs['manager_ranks'] = set([0])
        libE_specs['worker_ranks'] = set(range(1,MPI.COMM_WORLD.Get_size()))

    if 'color' not in libE_specs:
        libE_specs['color'] = 0

    assert isinstance(sim_specs,dict), "sim_specs must be a dictionary"
    assert isinstance(gen_specs,dict), "gen_specs must be a dictionary"
    assert isinstance(libE_specs,dict), "libE_specs must be a dictionary"
    assert isinstance(alloc_specs,dict), "alloc_specs must be a dictionary"
    assert isinstance(exit_criteria,dict), "exit_criteria must be a dictionary"
    assert isinstance(libE_specs['worker_ranks'],set), "libE_specs['worker_ranks'] must be a set"
    assert isinstance(libE_specs['manager_ranks'],set), "libE_specs['manager_ranks'] must be a set"

    assert len(exit_criteria)>0, "Must have some exit criterion"
    valid_term_fields = ['sim_max','gen_max','elapsed_wallclock_time','stop_val']
    assert any([term_field in exit_criteria for term_field in valid_term_fields]), "Must have a valid termination option: " + valid_term_fields 

    assert len(sim_specs['out']), "sim_specs must have 'out' entries"
    assert len(gen_specs['out']), "gen_specs must have 'out' entries"
    assert len(libE_specs['worker_ranks']), "Must have at least one worker rank"
    assert len(libE_specs['manager_ranks']), "Must have at least one manager rank"

    if 'stop_val' in exit_criteria:
        assert exit_criteria['stop_val'][0] in [e[0] for e in sim_specs['out']] + [e[0] for e in gen_specs['out']],\
               "Can't stop on " + exit_criteria['stop_val'][0] + " if it's not \
               returned from sim_specs['out'] or gen_specs['out']"
    
    if 'num_inst' in gen_specs and 'batch_mode' in gen_specs:
        assert gen_specs['num_inst'] <= 1 or not gen_specs['batch_mode'],\
               "Can't have more than one 'num_inst' for 'batch_mode' generator"

    from libE_fields import libE_fields

    if ('sim_id',int) in gen_specs['out'] and 'sim_id' in gen_specs['in']:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('\n' + 79*'*' + '\n'
                   "User generator script will be creating sim_id.\n"\
                   "Take care to do this sequentially.\n"\
                   "Also, any information given back for existing sim_id values will be overwritten!\n"\
                   "So everything in gen_out should be in gen_in!"\
                    '\n' + 79*'*' + '\n\n')
            sys.stdout.flush()
        libE_fields = libE_fields[1:] # Must remove 'sim_id' from libE_fields because it's in gen_specs['out']

    H = np.zeros(1 + len(H0), dtype=libE_fields + sim_specs['out'] + gen_specs['out']) 
    
    if len(H0):
        fields = H0.dtype.names
        assert set(fields).issubset(set(H.dtype.names)), "H0 contains fields not in H. Exiting"
        if 'returned' in fields:
            assert np.all(H0['returned']), "H0 contains unreturned points. Exiting"

        for field in fields:
            assert H[field].ndim == H0[field].ndim, "H0 and H have different ndim for field: " + field + ". Exiting"
            assert np.all(np.array(H[field].shape) >= np.array(H0[field].shape)), "H is not large enough to receive all of the components of H0 in field: " + field + ". Exiting"

    return libE_specs
