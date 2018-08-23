"""
Main libEnsemble routine
============================================

"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['libE']

from mpi4py import MPI
import numpy as np
import sys
import logging
import traceback

# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

# Set root logger
# (Set above libe imports so errors in import are captured)
# LEVEL: DEBUG/INFO/WARNING/ERROR
logging.basicConfig(level=logging.INFO, format='%(name)s (%(levelname)s): %(message)s')

from libensemble.history import History
from libensemble.libE_manager import manager_main
from libensemble.libE_worker import worker_main
from libensemble.calc_info import CalcInfo
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first

logger = logging.getLogger(__name__)
#For debug messages in this module  - uncomment (see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def comms_abort(comm):
    '''Abort all MPI ranks'''
    #This will be in comms module
    #comm arg will then be replaced with self.comm
    #Exit code 1 to represent an abort
    comm.Abort(1)    

  
def libE(sim_specs, gen_specs, exit_criteria, persis_info={},
         alloc_specs={'alloc_f': give_sim_work_first, 'out':[('allocated', bool)]},
         libE_specs={'comm': MPI.COMM_WORLD, 'color': 0}, H0=[]):
    """
    libE(sim_specs, gen_specs, exit_criteria, persis_info={}, alloc_specs={'alloc_f': give_sim_work_first, 'out':[('allocated',bool)]}, libE_specs={'comm': MPI.COMM_WORLD, 'color': 0}, H0 =[])

    This is the outer libEnsemble routine. If the rank in libE_specs['comm'] is
    0, manager_main is run. Otherwise, worker_main is run.
    (Some libEnsemble subroutines assume that the manager is always (only) rank 0.)

    Parameters
    ----------

    sim_specs: :obj:`dict`

        Specifications for the simulation function
        :doc:`(example)<data_structures/sim_specs>`
            

    gen_specs: :obj:`dict`

        Specifications for the generator function
        :doc:`(example)<data_structures/gen_specs>`

    exit_criteria: :obj:`dict`

        Tell libEnsemble when to stop a run
        :doc:`(example)<data_structures/exit_criteria>`

    persis_info: :obj:`dict`, optional

        Persistent information to be passed between user functions
        :doc:`(example)<data_structures/persis_info>`

    alloc_specs: :obj:`dict`, optional

        Specifications for the allocation function
        :doc:`(example)<data_structures/alloc_specs>`

    libE_specs: :obj:`dict`, optional

        Specifications for libEnsemble
        :doc:`(example)<data_structures/libE_specs>`

    H0: :obj:`dict`, optional

        A previous libEnsemble history to be prepended to the history in the
        current libEnsemble run
        :doc:`(example)<data_structures/history_array>`

    Returns
    -------

    H: :obj:`dict`

        History array storing rows for each point.  :doc:`(example)<data_structures/history_array>`
        Dictionary containing persistent info

    persis_info: :obj:`dict`

        Final state of persistent information
        :doc:`(example)<data_structures/persis_info>`

    exit_flag: :obj:`int`

        Flag containing job status: 0 = No errors, 2 = Manager timed out and ended simulation

    """
    #sys.excepthook = comms_abort(libE_specs['comm'])
    H = exit_flag = []
    libE_specs = check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    #create_exception = this_does_not_exist

    if libE_specs['comm'].Get_rank() == 0:   
        hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)        
        try:
            persis_info, exit_flag = manager_main(hist, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, persis_info)
        except Exception as e:
            
            # Put this in a manager_abort() routine. Then if worker exc - will signal mananger and if man receives abort tag (on iprobe)
            #- will
            # call abort routine also - worker can wait to give manager time - and then if nothing - mpi_abort itself.
            
            # Manager exceptions are fatal
            eprint("\nManager exception raised .. aborting ensemble:\n") #datetime
            eprint(traceback.format_exc())                            
            eprint("\nDumping ensemble with {} sims evaluated:\n".format(hist.sim_count)) #datetime  
            filename = 'libE_history_at_abort_' + str(hist.sim_count) + '.npy'
            np.save(filename,hist.trim_H())
            
            #Could have timing in here still...
            sys.stdout.flush()
            sys.stderr.flush()
            #sys.excepthook = comms_abort(libE_specs['comm'])
            comms_abort(libE_specs['comm'])
            #raise
                
        else:
            logger.debug("Manager exiting")
            print(libE_specs['comm'].Get_size(), exit_criteria)
            sys.stdout.flush()

    else: 
        try:
            worker_main(libE_specs, sim_specs, gen_specs)
        except Exception as e:
            # Worker exceptions fatal
            eprint("\nWorker exception raised on rank {} .. aborting ensemble:\n".format(libE_specs['comm'].Get_rank()))
            eprint(traceback.format_exc())
            sys.stdout.flush()
            sys.stderr.flush()
            #Cant dump hist from a worker unless keep a copy updated on workers (or do mpi_comm with manager here)
            #sys.excepthook = comms_abort(libE_specs['comm'])
            comms_abort(libE_specs['comm'])
            #raise
        else:
            logger.debug("Worker {} exiting".format(libE_specs['comm'].Get_rank()))

    # Create calc summary file
    libE_specs['comm'].Barrier()
    if libE_specs['comm'].Get_rank() == 0:
        CalcInfo.merge_statfiles()
        H = hist.trim_H()

    #return hist, persis_info, exit_flag
    #import pdb; pdb.set_trace()
    #currently return hist.H so dont need to modify calling scripts
    return H, persis_info, exit_flag



def check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0):
    """
    Check if the libEnsemble arguments are of the correct data type contain
    sufficient information to perform a run.
    """

    if 'comm' not in libE_specs:
        libE_specs['comm'] = MPI.COMM_WORLD

    if 'color' not in libE_specs:
        libE_specs['color'] = 0

    assert isinstance(sim_specs, dict), "sim_specs must be a dictionary"
    assert isinstance(gen_specs, dict), "gen_specs must be a dictionary"
    assert isinstance(libE_specs, dict), "libE_specs must be a dictionary"
    assert isinstance(alloc_specs, dict), "alloc_specs must be a dictionary"
    assert isinstance(exit_criteria, dict), "exit_criteria must be a dictionary"

    assert len(exit_criteria) > 0, "Must have some exit criterion"
    valid_term_fields = ['sim_max', 'gen_max', 'elapsed_wallclock_time', 'stop_val']
    assert any([term_field in exit_criteria for term_field in valid_term_fields]), "Must have a valid termination option: " + str(valid_term_fields)

    assert len(sim_specs['out']), "sim_specs must have 'out' entries"
    assert len(gen_specs['out']), "gen_specs must have 'out' entries"

    if 'stop_val' in exit_criteria:
        assert exit_criteria['stop_val'][0] in [e[0] for e in sim_specs['out']] + [e[0] for e in gen_specs['out']],\
               "Can't stop on " + exit_criteria['stop_val'][0] + " if it's not \
               returned from sim_specs['out'] or gen_specs['out']"

    if 'num_inst' in gen_specs and 'batch_mode' in gen_specs:
        assert gen_specs['num_inst'] <= 1 or not gen_specs['batch_mode'],\
               "Can't have more than one 'num_inst' for 'batch_mode' generator"

    from libensemble.libE_fields import libE_fields

    if ('sim_id', int) in gen_specs['out']:
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
        assert set(fields).issubset(set(H.dtype.names)), "H0 contains fields %r not in H. Exiting" % set(fields).difference(set(H.dtype.names)) 
        if 'returned' in fields:
            assert np.all(H0['returned']), "H0 contains unreturned points. Exiting"

        for field in fields:
            assert H[field].ndim == H0[field].ndim, "H0 and H have different ndim for field: " + field + ". Exiting"
            assert np.all(np.array(H[field].shape) >= np.array(H0[field].shape)), "H is not large enough to receive all of the components of H0 in field: " + field + ". Exiting"

    return libE_specs
