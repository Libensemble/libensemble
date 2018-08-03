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
    - gracefully pause simulation [sim]

"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from libensemble.libE_manager import manager_main
from libensemble.libE_worker import Worker, worker_main
from libensemble.calc_info import CalcInfo

from mpi4py import MPI
import numpy as np
import sys,os 
import traceback

# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def libE(sim_specs, gen_specs, exit_criteria, 
        alloc_specs={'alloc_f': give_sim_work_first, 'out':[('allocated',bool)]},
        libE_specs={'comm': MPI.COMM_WORLD, 'color': 0, 'manager_ranks': set([0]), 'worker_ranks': set(range(1,MPI.COMM_WORLD.Get_size()))}, 
        H0=[], persis_info={}):
    """ 
    libE(sim_specs, gen_specs, exit_criteria, alloc_specs={'alloc_f': give_sim_work_first, 'out':[('allocated',bool)]}, libE_specs={'comm': MPI.COMM_WORLD, 'color': 0, 'manager_ranks': set([0]), 'worker_ranks': set(range(1,MPI.COMM_WORLD.Get_size()))}, H0=[], persis_info={})
    
    This is the outer libEnsemble routine. It checks each rank in libE_specs['comm']
    against libE_specs['manager_ranks'] or libE_specs['worker_ranks'] and
    either runs manager_main or worker_main 
    (Some subroutines currently assume that the manager is always (only) rank 0.)
    
    Parameters
    ----------
    
    sim_specs: dict:
        :doc:`data_structures/sim_specs` for this ensemble
        
    gen_specs: dict:
        :doc:`data_structures/gen_specs` for this ensemble
            
    exit_criteria: dict: 
        :doc:`data_structures/exit_criteria` for this ensemble   
        
    alloc_specs: dict, optional :
        :doc:`data_structures/alloc_specs` for this ensemble   
        
    libE_specs: dict :
        :doc:`data_structures/libe_specs` for this ensemble       
    
    H0: :doc:`data_structures/history_array` : 
        A previous libEnsemble history to be prepended to the history in the current libEnsemble run.
        
    persis_info: dict :
        :doc:`data_structures/persis_info` for this ensemble      
    
    Returns
    -------

    H: :doc:`data_structures/history_array`:
        History array storing rows for each point.
        
    persis_info: Dict:
        :doc:`data_structures/persis_info`
    
    exit_flag: int:
        Flag containing job status: 0 = No errors, 2 = Manager timed out and ended simulation

    """
 
    H=exit_flag=[]
    libE_specs = check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    
    if libE_specs['comm'].Get_rank() in libE_specs['manager_ranks']:
        try:
            H, persis_info, exit_flag = manager_main(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0, persis_info)
        except Exception as e:
            # Manager exceptions are fatal
            eprint("\nManager exception raised .. aborting ensemble:\n") #datetime
            #Could have timing in here still...
            eprint(traceback.format_exc())
            sys.stdout.flush()
            sys.stderr.flush()
            # libE_specs['comm'].Abort()
        else:
            print(libE_specs['comm'].Get_size(),exit_criteria)
            sys.stdout.flush()

    else: #libE_specs['comm'].Get_rank() in libE_specs['worker_ranks']:        
        try:
            worker_main(libE_specs, sim_specs, gen_specs)
        except Exception as e:
            # Currently make worker exceptions fatal
            eprint("\nWorker exception raised on rank {} .. aborting ensemble:\n".format(libE_specs['comm'].Get_rank()))
            eprint(traceback.format_exc())
            sys.stdout.flush()
            sys.stderr.flush()
            # libE_specs['comm'].Abort()

    # Create calc summary file
    libE_specs['comm'].Barrier()
    if libE_specs['comm'].Get_rank() in libE_specs['manager_ranks']:
        CalcInfo.merge_statfiles()

    return H, persis_info, exit_flag



def check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0):
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

    assert libE_specs['comm'].Get_rank() in libE_specs['worker_ranks'] | libE_specs['manager_ranks'], \
            "The communicator has a rank that is not a worker and not a manager"
    assert isinstance(sim_specs,dict), "sim_specs must be a dictionary"
    assert isinstance(gen_specs,dict), "gen_specs must be a dictionary"
    assert isinstance(libE_specs,dict), "libE_specs must be a dictionary"
    assert isinstance(alloc_specs,dict), "alloc_specs must be a dictionary"
    assert isinstance(exit_criteria,dict), "exit_criteria must be a dictionary"
    assert isinstance(libE_specs['worker_ranks'],set), "libE_specs['worker_ranks'] must be a set"
    assert isinstance(libE_specs['manager_ranks'],set), "libE_specs['manager_ranks'] must be a set"

    assert len(exit_criteria)>0, "Must have some exit criterion"
    valid_term_fields = ['sim_max','gen_max','elapsed_wallclock_time','stop_val']
    assert any([term_field in exit_criteria for term_field in valid_term_fields]), "Must have a valid termination option: " + str(valid_term_fields)

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

    from libensemble.libE_fields import libE_fields

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
