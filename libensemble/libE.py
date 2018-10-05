"""
Main libEnsemble routine
============================================

"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['libE']

import sys
import logging
import traceback

import numpy as np

from libensemble.history import History
from libensemble.libE_manager import manager_main
from libensemble.libE_worker import worker_main
from libensemble.calc_info import CalcInfo
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.message_numbers import \
     EVAL_SIM_TAG, EVAL_GEN_TAG, ABORT_ENSEMBLE

logger = logging.getLogger(__name__)
#For debug messages in this module  - uncomment
#logger.setLevel(logging.DEBUG)


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def libE(sim_specs, gen_specs, exit_criteria,
         persis_info={},
         alloc_specs={'alloc_f': give_sim_work_first,
                      'out':[('allocated', bool)]},
         libE_specs={},
         H0=[]):
    """This is the outer libEnsemble routine.

    We dispatch to different types of worker teams depending on
    the contents of libE_specs.  If 'comm' is a field, we use MPI;
    if 'nthreads' is a field, we use threads; if 'nprocesses' is a
    field, we use multiprocessing.

    If an exception is encountered by the manager or workers, the
    history array is dumped to file and MPI abort is called.

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

        History array storing rows for each point.
        :doc:`(example)<data_structures/history_array>`
        Dictionary containing persistent info

    persis_info: :obj:`dict`

        Final state of persistent information
        :doc:`(example)<data_structures/persis_info>`

    exit_flag: :obj:`int`

        Flag containing job status: 0 = No errors,
        1 = Exception occured and MPI aborted,
        2 = Manager timed out and ended simulation
    """

    if 'nthreads' in libE_specs or 'nprocesses' in libE_specs:
        libE_f = libE_local
    else:
        libE_f = libE_mpi

    return libE_f(sim_specs, gen_specs, exit_criteria,
                  persis_info, alloc_specs, libE_specs, H0)


# ==================== MPI version =================================


def comms_abort(comm):
    '''Abort all MPI ranks'''
    comm.Abort(1) # Exit code 1 to represent an abort


def comms_signal_abort_to_man(comm):
    '''Worker signal manager to abort'''
    comm.send(obj=None, dest=0, tag=ABORT_ENSEMBLE)


def libE_mpi(sim_specs, gen_specs, exit_criteria,
             persis_info, alloc_specs, libE_specs, H0):
    "MPI version of the libE main routine"

    from mpi4py import MPI

    # Fill in default values (e.g. MPI_COMM_WORLD for communicator)
    if 'comm' not in libE_specs:
        libE_specs['comm'] = MPI.COMM_WORLD
    if 'color' not in libE_specs:
        libE_specs['color'] = 0

    comm = libE_specs['comm']
    is_master = (comm.Get_rank() == 0)

    # Set up logging to separate files (only if logging not already set)
    rank = comm.Get_rank()
    logging.basicConfig(filename='ensemble-{}.log'.format(rank),
                        level=logging.DEBUG,
                        format='%(name)s (%(levelname)s): %(message)s')

    # Check correctness of inputs
    libE_specs = check_inputs(is_master, libE_specs,
                              alloc_specs, sim_specs, gen_specs,
                              exit_criteria, H0)

    # Run manager or worker code, depending
    if is_master:
        return libE_mpi_manager(comm, sim_specs, gen_specs, exit_criteria,
                                persis_info, alloc_specs, libE_specs, H0)
    return libE_mpi_worker(comm, sim_specs, gen_specs, persis_info, libE_specs)


def libE_mpi_manager(mpi_comm, sim_specs, gen_specs, exit_criteria, persis_info,
                     alloc_specs, libE_specs, H0):
    "Manager routine run at rank 0."

    from libensemble.mpi_comms import MainMPIComm

    CalcInfo.make_statdir()
    mpi_comm.Barrier()

    exit_flag = []
    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)
    try:

        # Exchange dtypes
        mpi_comm.bcast(obj=hist.H[sim_specs['in']].dtype)
        mpi_comm.bcast(obj=hist.H[gen_specs['in']].dtype)
        wcomms = [MainMPIComm(mpi_comm, w) for w in
                  range(1, mpi_comm.Get_size())]
        persis_info, exit_flag = \
          manager_main(hist, libE_specs, alloc_specs, sim_specs, gen_specs,
                       exit_criteria, persis_info, wcomms)

    except Exception:
        eprint(traceback.format_exc())
        eprint("\nManager exception raised .. aborting ensemble:\n")
        eprint("\nDumping ensemble history with {} sims evaluated:\n".
               format(hist.sim_count))
        filename = 'libE_history_at_abort_' + str(hist.sim_count) + '.npy'
        np.save(filename, hist.trim_H())
        sys.stdout.flush()
        sys.stderr.flush()
        comms_abort(mpi_comm)
    else:
        logger.debug("Manager exiting")
        print(mpi_comm.Get_size(), exit_criteria)
        sys.stdout.flush()

    # Create calc summary file
    mpi_comm.Barrier()
    CalcInfo.merge_statfiles()

    H = hist.trim_H()
    return H, persis_info, exit_flag


def libE_mpi_worker(mpi_comm, sim_specs, gen_specs, persis_info, libE_specs):
    "Worker routine run at ranks > 0."

    from libensemble.mpi_comms import MainMPIComm
    mpi_comm.Barrier()
    try:
        # Exchange dtypes and set up comm
        dtypes = {EVAL_SIM_TAG: None, EVAL_GEN_TAG: None}
        dtypes[EVAL_SIM_TAG] = mpi_comm.bcast(dtypes[EVAL_SIM_TAG], root=0)
        dtypes[EVAL_GEN_TAG] = mpi_comm.bcast(dtypes[EVAL_GEN_TAG], root=0)
        comm = MainMPIComm(mpi_comm)
        worker_main(comm, dtypes, sim_specs, gen_specs)
    except Exception:
        eprint("\nWorker exception raised on rank {} .. aborting ensemble:\n".
               format(mpi_comm.Get_rank()))
        eprint(traceback.format_exc())
        sys.stdout.flush()
        sys.stderr.flush()
        comms_signal_abort_to_man(mpi_comm)
    else:
        logger.debug("Worker {} exiting".format(libE_specs['comm'].Get_rank()))
    mpi_comm.Barrier()

    H = exit_flag = []
    return H, persis_info, exit_flag


# ==================== Thread/process version =================================


def libE_local(sim_specs, gen_specs, exit_criteria,
               persis_info, alloc_specs, libE_specs, H0):
    "Main routine for thread/process launch of libE."

    from libensemble.comms import QCommProcess, QCommThread, Timeout

    # Set up if we are going to use
    if 'nthreads' in libE_specs:
        QCommTP = QCommThread
        has_terminate = False
        nworkers = libE_specs['nthreads']
        logging.basicConfig(
            filename='ensemble.log', level=logging.DEBUG,
            format='[%(threadName)s] %(name)s (%(levelname)s): %(message)s')
    else:
        QCommTP = QCommProcess
        has_terminate = True
        nworkers = libE_specs['nprocesses']
        # TODO: Add appropriate logging here

    libE_specs = check_inputs(True, libE_specs,
                              alloc_specs, sim_specs, gen_specs,
                              exit_criteria, H0)

    CalcInfo.make_statdir()

    exit_flag = []
    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Launch workers here
    dtypes = {EVAL_SIM_TAG: hist.H[sim_specs['in']].dtype,
              EVAL_GEN_TAG: hist.H[gen_specs['in']].dtype}

    try:
        wcomms = [QCommTP(worker_main, dtypes, sim_specs, gen_specs, w)
                  for w in range(1, nworkers+1)]
        for wcomm in wcomms:
            wcomm.run()
        persis_info, exit_flag = \
          manager_main(hist, libE_specs, alloc_specs, sim_specs, gen_specs,
                       exit_criteria, persis_info, wcomms)

    except Exception:
        eprint(traceback.format_exc())
        eprint("\nManager exception raised .. aborting ensemble:\n")
        eprint("\nDumping ensemble history with {} sims evaluated:\n".
               format(hist.sim_count))
        filename = 'libE_history_at_abort_' + str(hist.sim_count) + '.npy'
        np.save(filename, hist.trim_H())
        sys.stdout.flush()
        sys.stderr.flush()
    else:
        logger.debug("Manager exiting")
        print(nworkers, exit_criteria)
        sys.stdout.flush()

    # Join on workers here (and terminate forcefully if needed)
    for wcomm in wcomms:
        try:
            wcomm.result(timeout=libE_specs.get('worker_timeout'))
        except Timeout:
            if has_terminate:
                wcomm.terminate()

    # Create calc summary file
    CalcInfo.merge_statfiles()

    H = hist.trim_H()
    return H, persis_info, exit_flag


# ==================== Common input checking =================================


_USER_SIM_ID_WARNING = '\n' + 79*'*' + '\n' + \
"""User generator script will be creating sim_id.
Take care to do this sequentially.
Also, any information given back for existing sim_id values will be overwritten!
So everything in gen_out should be in gen_in!""" + \
'\n' + 79*'*' + '\n\n'


def check_consistent_field(name, field0, field1):
    "Check that new field (field1) is compatible with an old field (field0)."
    assert field0.ndim == field1.ndim, \
      "H0 and H have different ndim for field {}".format(name)
    assert (np.all(np.array(field1.shape) >= np.array(field0.shape))), \
      "H too small to receive all components of H0 in field {}".format(name)


def check_inputs(is_master, libE_specs, alloc_specs, sim_specs, gen_specs,
                 exit_criteria, H0):
    """
    Check if the libEnsemble arguments are of the correct data type contain
    sufficient information to perform a run.
    """

    # Check all the input fields are dicts
    assert isinstance(sim_specs, dict), "sim_specs must be a dictionary"
    assert isinstance(gen_specs, dict), "gen_specs must be a dictionary"
    assert isinstance(libE_specs, dict), "libE_specs must be a dictionary"
    assert isinstance(alloc_specs, dict), "alloc_specs must be a dictionary"
    assert isinstance(exit_criteria, dict), "exit_criteria must be a dictionary"

    # Check for at least one valid exit criterion
    assert len(exit_criteria) > 0, "Must have some exit criterion"
    valid_term_fields = ['sim_max', 'gen_max',
                         'elapsed_wallclock_time', 'stop_val']
    assert any([term_field in exit_criteria for
                term_field in valid_term_fields]), \
        "Must have a valid termination option: " + str(valid_term_fields)

    # Check that sim/gen have 'out' entries
    assert len(sim_specs['out']), "sim_specs must have 'out' entries"
    assert len(gen_specs['out']), "gen_specs must have 'out' entries"

    # If exit on stop, make sure it is something that a sim/gen outputs
    if 'stop_val' in exit_criteria:
        stop_name = exit_criteria['stop_val'][0]
        sim_out_names = [e[0] for e in sim_specs['out']]
        gen_out_names = [e[0] for e in gen_specs['out']]
        assert stop_name in sim_out_names + gen_out_names, \
          "Can't stop on {} if it's not in a sim/gen output".format(stop_name)

    # Handle if gen outputs sim IDs
    from libensemble.libE_fields import libE_fields
    if ('sim_id', int) in gen_specs['out']:
        if is_master:
            print(_USER_SIM_ID_WARNING)
            sys.stdout.flush()
         # Must remove 'sim_id' from libE_fields (it is in gen_specs['out'])
        libE_fields = libE_fields[1:]

    # Set up history -- combine libE_fields and sim/gen/alloc specs
    H = np.zeros(1 + len(H0),
                 dtype=libE_fields + list(set(sim_specs['out'] +
                                              gen_specs['out'] +
                                              alloc_specs.get('out', []))))

    # Sanity check prior history
    if len(H0):
        fields = H0.dtype.names

        # Prior history must contain the fields in new history
        assert set(fields).issubset(set(H.dtype.names)), \
          "H0 contains fields {} not in H.".\
          format(set(fields).difference(set(H.dtype.names)))

        # Prior history cannot contain unreturned points
        assert 'returned' not in fields or np.all(H0['returned']), \
          "H0 contains unreturned points."

        # Check dimensional compatibility of fields
        for field in fields:
            check_consistent_field(field, H0[field], H[field])

    return libE_specs
