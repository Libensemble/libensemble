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

def libE(sim_specs, gen_specs, exit_criteria, failure_processing={},
        alloc_specs={'alloc_f': give_sim_work_first, 'out':[('allocated',bool)]},
        libE_specs={'comm': MPI.COMM_WORLD, 'color': 0, 'manager_ranks': set([0]), 'worker_ranks': set(range(1,MPI.COMM_WORLD.Get_size()))}, 
        H0=[], persis_info={}):
    """ 
    libE(sim_specs, gen_specs, exit_criteria, failure_processing={}, alloc_specs={'alloc_f': give_sim_work_first, 'out':[('allocated',bool)]}, libE_specs={'comm': MPI.COMM_WORLD, 'color': 0, 'manager_ranks': set([0]), 'worker_ranks': set(range(1,MPI.COMM_WORLD.Get_size()))}, H0=[], persis_info={})
    
    This is the outer libEnsemble routine. It checks each rank in libE_specs['comm']
    against libE_specs['manager_ranks'] or libE_specs['worker_ranks'] and
    either runs manager_main or worker_main 
    (Some subroutines currently assume that the manager is always (only) rank 0.)
    
    Parameters
    ----------
    
    sim_specs: [dict]:

        Required keys :    
        
        'sim_f' [func] : 
            the simulation function being evaluated
        'in' [list] :
            field names (as strings) that will be given to sim_f
        'out' [list of tuples (field name, data type, [size])] :
            sim_f outputs that will be stored in the libEnsemble history
            
        Optional keys :
        
        'save_every_k' [int] :
            Save history array every k steps
        'sim_dir' [str] :
            Name of simulation directory which will be copied for each worker
        'sim_dir_prefix' [str] :
            A prefix path specifying where to create sim directories
        
        Additional entires in sim_specs will be given to sim_f
        
    gen_specs: [dict]:

        Required keys :     
        
        'gen_f' [func] : 
            generates inputs to sim_f
        'in' [list] : 
            field names (as strings) that will be given to gen_f
        'out' [list of tuples (field name, data type, [size])] :
            gen_f outputs that will be stored in the libEnsemble history
            
        Optional keys :
    
        'save_every_k' [int] :
            Save history array every k steps
        'queue_update_function' [func] :
            Additional entires in gen_specs will be given to gen_f
            
    exit_criteria: [dict]: 
        
        Optional keys (At least one must be given) :
        
        'sim_max' [int] : 
            Stop after this many sim_f evaluations have been completed
        'gen_max' [int] : 
            Stop after this many points have been generated by gen_f
        'elapsed_wallclock_time' [float] : 
            Stop after this amount of seconds have elapsed (since the libEnsemble manager has been initialized)
        'stop_val' [(str,float)] : 
            Stop when H[str] (for some field str returned from sim_out or gen_out) has been observed with a value less than the float given
        
    alloc_specs: [dict, optional] :
        'alloc_f' [func] :
            Default: give_sim_work_first
        'out' [list of tuples] :
            Default: [('allocated',bool)]
        'batch_mode' [bool] :
            Default: []
        'num_inst' [int] :
            Default: []
            
        The 'batch_mode' and 'num_inst' are specific arguments for the allocation function give_sim_work_first
    
    libE_specs [dict] :
        'comm' [MPI communicator] :
            libEnsemble communicator. Default: MPI.COMM_WORLD
        'color' [int] :
            Communicator color. Default: 0
        'manager_ranks' [set] :
            Default: [0]
        'worker_ranks' [set] :
            Default: [1 to comm.Get_size()-1]
        'queue_update_function' [func] :
            Default: []
    
    H0: numpy array: 
        A previous libEnsemble history to be prepended to the history in the current libEnsemble run.
        
    persis_info [dict] :
    
    Returns
    -------

    H: numpy structured array
        History array storing rows for each point. Field names are in
        libensemble/libE_fields.py 
        
    persis_info: Dict
        Dictionary containing persistent info
    
    exit_flag: int
        Flag containing job status: 0 = No errors, 2 = Manager timed out and ended simulation

    """
 
    libE_specs = check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, failure_processing, exit_criteria, H0)
    
    if libE_specs['comm'].Get_rank() in libE_specs['manager_ranks']:
        try:
            H, persis_info, exit_flag = manager_main(libE_specs, alloc_specs, sim_specs, gen_specs, failure_processing, exit_criteria, H0, persis_info)
        except Exception as e:
            # Manager exceptions are fatal
            eprint("\nManager exception raised .. aborting ensemble:\n") #datetime
            #Could have timing in here still...
            eprint(traceback.format_exc())
            sys.stdout.flush()
            sys.stderr.flush()
            libE_specs['comm'].Abort()
        else:
            print(libE_specs['comm'].Get_size(),exit_criteria)
            sys.stdout.flush()

    elif libE_specs['comm'].Get_rank() in libE_specs['worker_ranks']:        
        try:
            worker_main(libE_specs, sim_specs, gen_specs); H=exit_flag=[]
        except Exception as e:
            # Currently make worker exceptions fatal
            eprint("\nWorker exception raised on rank {} .. aborting ensemble:\n".format(libE_specs['comm'].Get_rank()))
            eprint(traceback.format_exc())
            sys.stdout.flush()
            sys.stderr.flush()
            libE_specs['comm'].Abort()
    else:
        print("Rank: %d not manager or worker" % libE_specs['comm'].Get_rank()); H=exit_flag=[]

    # Create calc summary file
    libE_specs['comm'].Barrier()
    CalcInfo.merge_statfiles()

    return H, persis_info, exit_flag



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
