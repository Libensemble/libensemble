"""
Main libEnsemble routine
============================================

"""

__all__ = ['libE']

import os
import sys
import logging
import traceback
import random
import socket
import pickle  # Only used when saving output on error

import numpy as np
import libensemble.util.launcher as launcher
from libensemble.util.timer import Timer
from libensemble.history import History
from libensemble.libE_manager import manager_main, ManagerException
from libensemble.libE_worker import worker_main
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.comms.comms import QCommProcess, Timeout
from libensemble.comms.logs import manager_logging_config
from libensemble.comms.tcp_mgr import ServerQCommManager, ClientQCommManager

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def report_manager_exception(hist, persis_info, mgr_exc=None):
    "Write out exception manager exception to stderr and flush streams."
    if mgr_exc is not None:
        from_line, msg, exc = mgr_exc.args
        eprint("\n---- {} ----".format(from_line))
        eprint("Message: {}".format(msg))
        eprint(exc)
        eprint("--------\n")
    else:
        eprint(traceback.format_exc())
    eprint("\nManager exception raised .. aborting ensemble:\n")
    eprint("\nDumping ensemble history with {} sims evaluated:\n".
           format(hist.sim_count))

    filename = 'libE_history_at_abort_' + str(hist.sim_count)
    np.save(filename + '.npy', hist.trim_H())
    with open(filename + '.pickle', "wb") as f:
        pickle.dump(persis_info, f)

    sys.stdout.flush()
    sys.stderr.flush()


def libE(sim_specs, gen_specs, exit_criteria,
         persis_info={},
         alloc_specs={'alloc_f': give_sim_work_first,
                      'out': [('allocated', bool)]},
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
        1 = Exception occured
        2 = Manager timed out and ended simulation
    """

    comms_type = libE_specs.get('comms', 'mpi')
    libE_funcs = {'mpi': libE_mpi,
                  'tcp': libE_tcp,
                  'local': libE_local}
    assert comms_type in libE_funcs, "Unknown comms type: {}".format(comms_type)
    return libE_funcs[comms_type](sim_specs, gen_specs, exit_criteria,
                                  persis_info, alloc_specs, libE_specs, H0)


def libE_manager(wcomms, sim_specs, gen_specs, exit_criteria, persis_info,
                 alloc_specs, libE_specs, hist,
                 on_abort=None, on_cleanup=None):
    "Generic manager routine run."

    if 'out' in gen_specs and ('sim_id', int) in gen_specs['out']:
        print(_USER_SIM_ID_WARNING)
        sys.stdout.flush()

    try:
        persis_info, exit_flag = \
            manager_main(hist, libE_specs, alloc_specs, sim_specs, gen_specs,
                         exit_criteria, persis_info, wcomms)
    except ManagerException as e:
        report_manager_exception(hist, persis_info, e)
        if libE_specs.get('abort_on_exception', True) and on_abort is not None:
            on_abort()
        raise
    except Exception:
        report_manager_exception(hist, persis_info)
        if libE_specs.get('abort_on_exception', True) and on_abort is not None:
            on_abort()
        raise
    else:
        logger.debug("Manager exiting")
        print(len(wcomms), exit_criteria)
        sys.stdout.flush()
    finally:
        if on_cleanup is not None:
            on_cleanup()

    H = hist.trim_H()
    return H, persis_info, exit_flag


# ==================== MPI version =================================


def comms_abort(comm):
    "Abort all MPI ranks"
    comm.Abort(1)  # Exit code 1 to represent an abort


def libE_mpi_defaults(libE_specs):
    "Fill in default values for MPI-based communicators."

    from mpi4py import MPI

    if 'comm' not in libE_specs:
        libE_specs['comm'] = MPI.COMM_WORLD
    if 'color' not in libE_specs:
        libE_specs['color'] = 0
    return libE_specs


def libE_mpi(sim_specs, gen_specs, exit_criteria,
             persis_info, alloc_specs, libE_specs, H0):
    "MPI version of the libE main routine"

    libE_specs = libE_mpi_defaults(libE_specs)
    comm = libE_specs['comm']
    rank = comm.Get_rank()
    is_master = (rank == 0)

    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Run manager or worker code, depending
    if is_master:
        return libE_mpi_manager(comm, sim_specs, gen_specs, exit_criteria,
                                persis_info, alloc_specs, libE_specs, H0)

    # Worker returns a subset of MPI output
    libE_mpi_worker(comm, sim_specs, gen_specs, libE_specs)
    return [], persis_info, []


def libE_mpi_manager(mpi_comm, sim_specs, gen_specs, exit_criteria, persis_info,
                     alloc_specs, libE_specs, H0):
    "Manager routine run at rank 0."

    from libensemble.comms.mpi import MainMPIComm

    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Lauch worker team
    wcomms = [MainMPIComm(mpi_comm, w) for w in
              range(1, mpi_comm.Get_size())]
    manager_logging_config()

    # Set up abort handler
    def on_abort():
        "Shut down MPI on error."
        comms_abort(mpi_comm)

    # Run generic manager
    return libE_manager(wcomms, sim_specs, gen_specs, exit_criteria,
                        persis_info, alloc_specs, libE_specs, hist,
                        on_abort=on_abort)


def libE_mpi_worker(mpi_comm, sim_specs, gen_specs, libE_specs):
    "Worker routine run at ranks > 0."

    from libensemble.comms.mpi import MainMPIComm
    comm = MainMPIComm(mpi_comm)
    worker_main(comm, sim_specs, gen_specs, log_comm=True)
    logger.debug("Worker {} exiting".format(libE_specs['comm'].Get_rank()))


# ==================== Process version =================================
def start_proc_team(nworkers, sim_specs, gen_specs, log_comm=True):
    "Launch a process worker team."
    wcomms = [QCommProcess(worker_main, sim_specs, gen_specs, w, log_comm)
              for w in range(1, nworkers+1)]
    for wcomm in wcomms:
        wcomm.run()
    return wcomms


def kill_proc_team(wcomms, timeout):
    "Join on workers (and terminate forcefully if needed)."
    for wcomm in wcomms:
        try:
            wcomm.result(timeout=timeout)
        except Timeout:
            wcomm.terminate()


def libE_local(sim_specs, gen_specs, exit_criteria,
               persis_info, alloc_specs, libE_specs, H0):
    "Main routine for thread/process launch of libE."

    nworkers = libE_specs['nprocesses']
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Launch worker team and set up logger
    wcomms = start_proc_team(nworkers, sim_specs, gen_specs)
    manager_logging_config()

    # Set up cleanup routine to shut down worker team
    def cleanup():
        "Handler to clean up comms team."
        kill_proc_team(wcomms, timeout=libE_specs.get('worker_timeout'))

    # Run generic manager
    return libE_manager(wcomms, sim_specs, gen_specs, exit_criteria,
                        persis_info, alloc_specs, libE_specs, hist,
                        on_cleanup=cleanup)


# ==================== TCP version =================================


def get_ip():
    "Get the IP address of the current host"
    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        return 'localhost'


def libE_tcp_authkey():
    "Generate an authkey if not assigned by manager."
    nonce = random.randrange(99999)
    return 'libE_auth_{}'.format(nonce)


def libE_tcp_default_ID():
    "Assign a (we hope unique) worker ID if not assigned by manager."
    return "{}_pid{}".format(get_ip(), os.getpid())


def libE_tcp(sim_specs, gen_specs, exit_criteria,
             persis_info, alloc_specs, libE_specs, H0):
    "Main routine for TCP multiprocessing launch of libE."

    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    if 'workerID' in libE_specs:
        libE_tcp_worker(sim_specs, gen_specs, libE_specs)
        return [], persis_info, []

    return libE_tcp_mgr(sim_specs, gen_specs, exit_criteria,
                        persis_info, alloc_specs, libE_specs, H0)


def libE_tcp_worker_launcher(libE_specs):
    "Get a launch function from libE_specs."
    if 'worker_launcher' in libE_specs:
        worker_launcher = libE_specs['worker_launcher']
    else:
        worker_cmd = libE_specs['worker_cmd']

        def worker_launcher(specs):
            "Basic worker launch function."
            return launcher.launch(worker_cmd, specs)
    return worker_launcher


def libE_tcp_start_team(manager, nworkers, workers,
                        ip, port, authkey, launchf):
    "Launch nworkers workers that attach back to a managers server."
    worker_procs = []
    specs = {'manager_ip': ip, 'manager_port': port, 'authkey': authkey}
    with Timer() as timer:
        for w in range(1, nworkers+1):
            logger.info("Manager is launching worker {}".format(w))
            if workers is not None:
                specs['worker_ip'] = workers[w-1]
                specs['tunnel_port'] = 0x71BE
            specs['workerID'] = w
            worker_procs.append(launchf(specs))
        logger.info("Manager is awaiting {} workers".format(nworkers))
        wcomms = manager.await_workers(nworkers)
        logger.info("Manager connected to {} workers ({} s)".
                    format(nworkers, timer.elapsed))
    return worker_procs, wcomms


def libE_tcp_mgr(sim_specs, gen_specs, exit_criteria,
                 persis_info, alloc_specs, libE_specs, H0):
    "Main routine for TCP multiprocessing launch of libE at manager."

    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Set up a worker launcher
    launchf = libE_tcp_worker_launcher(libE_specs)

    # Get worker launch parameters and fill in defaults for TCP/IP conn
    if 'nprocesses' in libE_specs:
        workers = None
        nworkers = libE_specs['nprocesses']
    elif 'workers' in libE_specs:
        workers = libE_specs['workers']
        nworkers = len(workers)
    ip = libE_specs.get('ip', None) or get_ip()
    port = libE_specs.get('port', 0)
    authkey = libE_specs.get('authkey', libE_tcp_authkey())

    with ServerQCommManager(port, authkey.encode('utf-8')) as manager:

        # Get port if needed because of auto-assignment
        if port == 0:
            _, port = manager.address

        manager_logging_config()
        logger.info("Launched server at ({}, {})".format(ip, port))

        # Launch worker team and set up logger
        worker_procs, wcomms =\
            libE_tcp_start_team(manager, nworkers, workers,
                                ip, port, authkey, launchf)

        def cleanup():
            "Handler to clean up launched team."
            for wp in worker_procs:
                launcher.cancel(wp, timeout=libE_specs.get('worker_timeout'))

        # Run generic manager
        return libE_manager(wcomms, sim_specs, gen_specs, exit_criteria,
                            persis_info, alloc_specs, libE_specs, hist,
                            on_cleanup=cleanup)


def libE_tcp_worker(sim_specs, gen_specs, libE_specs):
    "Main routine for TCP worker launched by libE."

    ip = libE_specs['ip']
    port = libE_specs['port']
    authkey = libE_specs['authkey']
    workerID = libE_specs['workerID']

    with ClientQCommManager(ip, port, authkey, workerID) as comm:
        worker_main(comm, sim_specs, gen_specs,
                    workerID=workerID, log_comm=True)
        logger.debug("Worker {} exiting".format(workerID))


# ==================== Common input checking =================================
_USER_SIM_ID_WARNING = \
    ('\n' + 79*'*' + '\n' +
     "User generator script will be creating sim_id.\n" +
     "Take care to do this sequentially.\n" +
     "Also, any information given back for existing sim_id values will be overwritten!\n" +
     "So everything in gen_specs['out'] should be in gen_specs['in']!" +
     '\n' + 79*'*' + '\n\n')


def check_consistent_field(name, field0, field1):
    "Check that new field (field1) is compatible with an old field (field0)."
    assert field0.ndim == field1.ndim, \
        "H0 and H have different ndim for field {}".format(name)
    assert (np.all(np.array(field1.shape) >= np.array(field0.shape))), \
        "H too small to receive all components of H0 in field {}".format(name)


def check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0):
    """
    Check if the libEnsemble arguments are of the correct data type contain
    sufficient information to perform a run.
    """

    if libE_specs.get('comms', 'undefined') in ['mpi']:
        assert libE_specs['comm'].Get_size() > 1, "Manager only - must be at least one worker (2 MPI tasks)"

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
    assert all([term_field in valid_term_fields for term_field in exit_criteria]), \
        "Valid termination options: " + str(valid_term_fields)

    # Check that sim/gen have 'out' entries
    assert len(sim_specs['out']), "sim_specs must have 'out' entries"
    assert not bool(gen_specs) or len(gen_specs['out']), "gen_specs must have 'out' entries"

    # If exit on stop, make sure it is something that a sim/gen outputs
    if 'stop_val' in exit_criteria:
        stop_name = exit_criteria['stop_val'][0]
        sim_out_names = [e[0] for e in sim_specs['out']]
        gen_out_names = [e[0] for e in gen_specs['out']]
        assert stop_name in sim_out_names + gen_out_names, \
            "Can't stop on {} if it's not in a sim/gen output".format(stop_name)

    # Sanity check prior history
    if len(H0):
        # Handle if gen outputs sim IDs
        from libensemble.libE_fields import libE_fields

        # Set up dummy history to see if it agrees with H0
        Dummy_H = np.zeros(1 + len(H0), dtype=libE_fields + list(set(sum([k['out'] for k in [sim_specs, alloc_specs, gen_specs] if k], []))))  # Combines all 'out' fields (if they exist) in sim_specs, gen_specs, or alloc_specs

        fields = H0.dtype.names

        # Prior history must contain the fields in new history
        assert set(fields).issubset(set(Dummy_H.dtype.names)), \
            "H0 contains fields {} not in the History.".\
            format(set(fields).difference(set(Dummy_H.dtype.names)))

        # Prior history cannot contain unreturned points
        assert 'returned' not in fields or np.all(H0['returned']), \
            "H0 contains unreturned points."

        # Check dimensional compatibility of fields
        for field in fields:
            check_consistent_field(field, H0[field], Dummy_H[field])
