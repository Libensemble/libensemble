"""
Main libEnsemble routine
============================================

"""

__all__ = ['libE']

import os
import logging
import random
import socket

import libensemble.util.launcher as launcher
from libensemble.util.timer import Timer
from libensemble.history import History
from libensemble.libE_manager import manager_main, ManagerException
from libensemble.libE_worker import worker_main
from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.comms.comms import QCommProcess, Timeout
from libensemble.comms.logs import manager_logging_config
from libensemble.comms.tcp_mgr import ServerQCommManager, ClientQCommManager
from libensemble.controller import JobController
from libensemble.utils import check_inputs, _USER_SIM_ID_WARNING, report_manager_exception

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


def libE(sim_specs, gen_specs, exit_criteria,
         persis_info={},
         alloc_specs={'alloc_f': give_sim_work_first,
                      'out': [('allocated', bool)],
                      'user': {'batch_mode': True}},
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

    H0: :obj:`numpy structured array`, optional

        A previous libEnsemble history to be prepended to the history in the
        current libEnsemble run
        :doc:`(example)<data_structures/history_array>`

    Returns
    -------

    H: :obj:`numpy structured array`

        History array storing rows for each point.
        :doc:`(example)<data_structures/history_array>`

    persis_info: :obj:`dict`

        Final state of persistent information
        :doc:`(example)<data_structures/persis_info>`

    exit_flag: :obj:`int`

        Flag containing final job status:

        0 = No errors
        1 = Exception occured
        2 = Manager timed out and ended simulation
        3 = Current process is not in libEnsemble MPI communicator
    """

    # Set default comms
    if 'comms' not in libE_specs:
        libE_specs['comms'] = 'mpi'

    libE_funcs = {'mpi': libE_mpi,
                  'tcp': libE_tcp,
                  'local': libE_local}

    comms_type = libE_specs.get('comms')

    assert comms_type in libE_funcs, "Unknown comms type: {}".format(comms_type)
    return libE_funcs[comms_type](sim_specs, gen_specs, exit_criteria,
                                  persis_info, alloc_specs, libE_specs, H0)


def libE_manager(wcomms, sim_specs, gen_specs, exit_criteria, persis_info,
                 alloc_specs, libE_specs, hist,
                 on_abort=None, on_cleanup=None):
    "Generic manager routine run."

    if 'out' in gen_specs and ('sim_id', int) in gen_specs['out']:
        logger.manager_warning(_USER_SIM_ID_WARNING)

    try:
        persis_info, exit_flag, elapsed_time = \
            manager_main(hist, libE_specs, alloc_specs, sim_specs, gen_specs,
                         exit_criteria, persis_info, wcomms)
        logger.info("libE_manager total time: {}".format(elapsed_time))

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
        logger.debug("Exiting with {} workers.".format(len(wcomms)))
        logger.debug("Exiting with exit criteria: {}".format(exit_criteria))
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
        libE_specs['comm'] = MPI.COMM_WORLD.Dup()
    return libE_specs, MPI.COMM_NULL


def libE_mpi(sim_specs, gen_specs, exit_criteria,
             persis_info, alloc_specs, libE_specs, H0):
    "MPI version of the libE main routine"

    libE_specs, mpi_comm_null = libE_mpi_defaults(libE_specs)
    comm = libE_specs['comm']

    if comm == mpi_comm_null:
        return [], persis_info, 3  # Process not in comm

    rank = comm.Get_rank()
    is_master = (rank == 0)
    check_inputs(libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    jobctl = JobController.controller
    if jobctl is not None:
        local_host = socket.gethostname()
        libE_nodes = set(comm.allgather(local_host))
        jobctl.add_comm_info(libE_nodes=libE_nodes, serial_setup=is_master)

    # Run manager or worker code, depending
    if is_master:
        return libE_mpi_manager(comm, sim_specs, gen_specs, exit_criteria,
                                persis_info, alloc_specs, libE_specs, H0)

    # Worker returns a subset of MPI output
    libE_mpi_worker(sim_specs, gen_specs, libE_specs)
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


def libE_mpi_worker(sim_specs, gen_specs, libE_specs):
    "Worker routine run at ranks > 0."

    from libensemble.comms.mpi import MainMPIComm
    comm = MainMPIComm(libE_specs['comm'])
    worker_main(comm, sim_specs, gen_specs, libE_specs, log_comm=True)
    logger.debug("Worker {} exiting".format(libE_specs['comm'].Get_rank()))


# ==================== Process version =================================
def start_proc_team(nworkers, sim_specs, gen_specs, libE_specs, log_comm=True):
    "Launch a process worker team."
    wcomms = [QCommProcess(worker_main, sim_specs, gen_specs, libE_specs, w, log_comm)
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

    jobctl = JobController.controller
    if jobctl is not None:
        local_host = socket.gethostname()
        jobctl.add_comm_info(libE_nodes=local_host, serial_setup=True)

    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Launch worker team and set up logger
    wcomms = start_proc_team(nworkers, sim_specs, gen_specs, libE_specs)
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

    is_worker = True if 'workerID' in libE_specs else False

    jobctl = JobController.controller
    if jobctl is not None:
        local_host = socket.gethostname()
        # TCP does not currently support auto_resources but when does, assume
        # each TCP worker is in a different resource pool (only knowing local_host)
        jobctl.add_comm_info(libE_nodes=local_host, serial_setup=not is_worker)

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
        worker_main(comm, sim_specs, gen_specs, libE_specs,
                    workerID=workerID, log_comm=True)
        logger.debug("Worker {} exiting".format(workerID))
