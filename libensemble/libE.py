"""
The libE module is the outer libEnsemble routine.

This module sets up the manager and the team of workers, configured according
to the contents of :ref:`libE_specs<datastruct-libe-specs>`. The manager/worker
communications scheme used in libEnsemble is parsed from the ``comms`` key if
present, with valid values being ``mpi``, ``local`` (for multiprocessing), or
``tcp``.

MPI is the default if ``nworkers`` is not given. However, if ``libE_specs["nworkers"]``
is specified, then ``local`` comms will be used unless a parallel MPI environment
is detected.

For ``mpi`` comms, if a communicator is specified, each call to this module
will initiate manager/worker communications on a duplicate of that
communicator. Otherwise, a duplicate of ``COMM_WORLD`` will be used.

In the vast majority of cases, programming with libEnsemble involves the creation
of a *calling script*, a Python file where libEnsemble is parameterized via
the various specification dictionaries (e.g. :ref:`libE_specs<datastruct-libe-specs>`,
:ref:`sim_specs<datastruct-sim-specs>`, and :ref:`gen_specs<datastruct-gen-specs>`).
The outer libEnsemble routine :meth:`libE()<libensemble.libE.libE>` is imported and
called with such dictionaries to initiate libEnsemble. A simple calling script
(from :doc:`the first tutorial<tutorials/local_sine_tutorial>`) may resemble:

.. code-block:: python
    :linenos:

    import numpy as np
    from libensemble.libE import libE
    from generator import gen_random_sample
    from simulator import sim_find_sine
    from libensemble.tools import add_unique_random_streams

    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["save_every_k_gens"] = 20

    gen_specs = {
        "gen_f": gen_random_sample,
        "out": [("x", float, (1,))],
        "user": {"lower": np.array([-3]), "upper": np.array([3]), "gen_batch_size": 5},
    }

    sim_specs = {"sim_f": sim_find_sine, "in": ["x"], "out": [("y", float)]}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 80}

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

This will initiate libEnsemble with a Manager and ``nworkers`` workers (parsed from
the command line), and runs on laptops or supercomputers. If an exception is
encountered by the manager or workers, the history array is dumped to file, and
MPI abort is called.

On macOS (since Python 3.8) and Windows, the default multiprocessing start method is ``"spawn"``
and you must place most calling script code (or just ``libE()`` / ``Ensemble().run()`` at a minimum) in
an ``if __name__ == "__main__:"`` block.

Therefore a calling script that is universal across
all platforms and comms-types may resemble:

.. code-block:: python
    :linenos:

    import numpy as np
    from libensemble.libE import libE
    from generator import gen_random_sample
    from simulator import sim_find_sine
    from libensemble.tools import add_unique_random_streams

    if __name__ == "__main__":
        nworkers, is_manager, libE_specs, _ = parse_args()

        libE_specs["save_every_k_gens"] = 20

        gen_specs = {
            "gen_f": gen_random_sample,
            "out": [("x", float, (1,))],
            "user": {
                "lower": np.array([-3]),
                "upper": np.array([3]),
                "gen_batch_size": 5,
            },
        }

        sim_specs = {
            "sim_f": sim_find_sine,
            "in": ["x"],
            "out": [("y", float)],
        }

        persis_info = add_unique_random_streams({}, nworkers + 1)

        exit_criteria = {"sim_max": 80}

        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

Alternatively, you may set the multiprocessing start method to ``"fork"`` via the following:

.. code-block:: python
    :linenos:

    from multiprocessing import set_start_method

    set_start_method("fork")

But note that this is incompatible with some libraries.

See below for the complete traditional API.
"""

__all__ = ["libE"]

import logging
import os
import pickle  # Only used when saving output on error
import socket
import sys
import traceback
from pathlib import Path

import numpy as np

from libensemble.comms.comms import QCommProcess, QCommThread, Timeout
from libensemble.comms.logs import manager_logging_config
from libensemble.comms.tcp_mgr import ClientQCommManager, ServerQCommManager
from libensemble.executors.executor import Executor
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.history import History
from libensemble.manager import LoggedException, WorkerException, manager_main, report_worker_exc
from libensemble.resources.platforms import get_platform
from libensemble.resources.resources import Resources
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs, _EnsembleSpecs
from libensemble.tools.alloc_support import AllocSupport
from libensemble.tools.tools import _USER_SIM_ID_WARNING
from libensemble.utils import launcher
from libensemble.utils.misc import specs_dump
from libensemble.utils.pydantic_bindings import libE_wrapper
from libensemble.utils.timer import Timer
from libensemble.version import __version__
from libensemble.worker import worker_main

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


@libE_wrapper
def libE(
    sim_specs: SimSpecs,
    gen_specs: GenSpecs,
    exit_criteria: ExitCriteria,
    persis_info: dict = {},
    alloc_specs: AllocSpecs = AllocSpecs(),
    libE_specs: LibeSpecs = {},
    H0=None,
) -> (np.ndarray, dict, int):
    """
    Parameters
    ----------

    sim_specs: :obj:`dict` or :class:`SimSpecs<libensemble.specs.SimSpecs>`

        Specifications for the simulation function
        :doc:`(example)<data_structures/sim_specs>`

    gen_specs: :obj:`dict` or :class:`GenSpecs<libensemble.specs.GenSpecs>`, Optional

        Specifications for the generator function
        :doc:`(example)<data_structures/gen_specs>`

    exit_criteria: :obj:`dict` or :class:`ExitCriteria<libensemble.specs.ExitCriteria>`, Optional

        Tell libEnsemble when to stop a run
        :doc:`(example)<data_structures/exit_criteria>`

    persis_info: :obj:`dict`, Optional

        Persistent information to be passed between user functions
        :doc:`(example)<data_structures/persis_info>`

    alloc_specs: :obj:`dict` or :class:`AllocSpecs<libensemble.specs.AllocSpecs>`, Optional

        Specifications for the allocation function
        :doc:`(example)<data_structures/alloc_specs>`

    libE_specs: :obj:`dict` or :class:`LibeSpecs<libensemble.specs.LibeSpecs>`, Optional

        Specifications for libEnsemble
        :doc:`(example)<data_structures/libE_specs>`

    H0: `NumPy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_, Optional

        A libEnsemble history to be prepended to this run's history
        :ref:`(example)<funcguides-history>`

    Returns
    -------

    H: `NumPy structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_

        History array storing rows for each point.
        :ref:`(example)<funcguides-history>`

    persis_info: :obj:`dict`

        Final state of persistent information
        :doc:`(example)<data_structures/persis_info>`

    exit_flag: :obj:`int`

        Flag containing final task status

        .. code-block::

            0 = No errors
            1 = Exception occurred
            2 = Manager timed out and ended simulation
            3 = Current process is not in libEnsemble MPI communicator
    """

    if H0 is None:
        H0 = np.empty([0])

    # check *everything*
    ensemble = _EnsembleSpecs(
        H0=H0,
        libE_specs=libE_specs,
        persis_info=persis_info,
        sim_specs=sim_specs,
        gen_specs=gen_specs,
        alloc_specs=alloc_specs,
        exit_criteria=exit_criteria,
    )

    (sim_specs, gen_specs, alloc_specs, libE_specs) = [
        specs_dump(spec, by_alias=True)
        for spec in [ensemble.sim_specs, ensemble.gen_specs, ensemble.alloc_specs, ensemble.libE_specs]
    ]
    exit_criteria = specs_dump(ensemble.exit_criteria, by_alias=True, exclude_none=True)

    # Extract platform info from settings or environment
    platform_info = get_platform(libE_specs)

    if libE_specs["dry_run"]:
        logger.manager_warning("Dry run. All libE() inputs validated. Exiting.")
        sys.exit()

    libE_funcs = {"mpi": libE_mpi, "tcp": libE_tcp, "local": libE_local, "threads": libE_local}

    Resources.init_resources(libE_specs, platform_info)
    if Executor.executor is not None:
        Executor.executor.add_platform_info(platform_info)

    # Reset gen counter.
    AllocSupport.gen_counter = 0

    return libE_funcs[libE_specs.get("comms", "mpi")](
        sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0
    )


def manager(
    wcomms,
    sim_specs,
    gen_specs,
    exit_criteria,
    persis_info,
    alloc_specs,
    libE_specs,
    hist: History,
    on_abort=None,
    on_cleanup=None,
):
    """Generic manager routine run."""
    logger.info("Logger initializing: [workerID] precedes each line. [0] = Manager")
    logger.info(f"libE version v{__version__}")

    if "out" in gen_specs and ("sim_id", int) in gen_specs["out"]:
        if hasattr(gen_specs["gen_f"], "__module__") and "libensemble.gen_funcs" not in gen_specs["gen_f"].__module__:
            logger.manager_warning(_USER_SIM_ID_WARNING)

    try:
        try:
            persis_info, exit_flag, elapsed_time = manager_main(
                hist, libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria, persis_info, wcomms
            )
            logger.info(f"Manager total time: {elapsed_time}")
        except LoggedException:
            # Exception already logged in manager
            raise
        except WorkerException as e:
            report_worker_exc(e)
            raise LoggedException(e.args[0], e.args[1]) from None
        except Exception as e:
            logger.error(traceback.format_exc())
            raise LoggedException(e.args) from None
    except Exception as e:
        exit_flag = 1  # Only exits if no abort/raise
        _dump_on_abort(
            hist, persis_info, save_H=libE_specs["save_H_and_persis_on_abort"], path=libE_specs.get("workflow_dir_path")
        )
        if libE_specs["abort_on_exception"] and on_abort is not None:
            on_cleanup()
            on_abort()
        raise LoggedException(*e.args, "See error details above and in ensemble.log") from None
    else:
        logger.debug("Manager exiting")
        logger.debug(f"Exiting with {len(wcomms)} workers.")
        logger.debug(f"Exiting with exit criteria: {exit_criteria}")
    finally:
        if on_cleanup is not None:
            on_cleanup()

    H = hist.trim_H()
    return H, persis_info, exit_flag


# ==================== MPI version =================================


class DupComm:
    """Duplicate MPI communicator for use with a with statement"""

    def __init__(self, mpi_comm):
        self.parent_comm = mpi_comm

    def __enter__(self):
        self.dup_comm = self.parent_comm.Dup()
        return self.dup_comm

    def __exit__(self, etype, value, traceback):
        self.dup_comm.Free()


def comms_abort(mpi_comm):
    """Abort all MPI ranks"""
    mpi_comm.Abort(1)  # Exit code 1 to represent an abort


def libE_mpi(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0):
    """MPI version of the libE main routine"""
    from mpi4py import MPI

    if libE_specs.get("mpi_comm") is None:
        libE_specs["mpi_comm"] = MPI.COMM_WORLD  # Will be duplicated immediately

    if libE_specs["mpi_comm"] == MPI.COMM_NULL:
        logger.manager_warning("*WARNING* libEnsemble detected a NULL communicator")
        return [], persis_info, 3  # Process not in mpi_comm

    assert libE_specs["mpi_comm"].Get_size() > 1, "Manager only - must be at least one worker (2 MPI tasks)"

    with DupComm(libE_specs["mpi_comm"]) as mpi_comm:
        is_manager = mpi_comm.Get_rank() == 0

        resources = Resources.resources
        if resources is not None:
            local_host = socket.gethostname()
            libE_nodes = list(set(mpi_comm.allgather(local_host)))
            resources.add_comm_info(libE_nodes=libE_nodes)
            nworkers = mpi_comm.Get_size() - 1

        exctr = Executor.executor
        if exctr is not None:
            exctr.set_resources(resources)
            if is_manager:
                exctr.serial_setup()

        # Run manager or worker code, depending
        if is_manager:
            if resources is not None:
                resources.set_resource_manager(nworkers)
            return libE_mpi_manager(
                mpi_comm, sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0
            )

        # Worker returns a subset of MPI output
        return libE_mpi_worker(mpi_comm, sim_specs, gen_specs, libE_specs)


def libE_mpi_manager(mpi_comm, sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0):
    """Manager routine runs on rank 0."""
    from libensemble.comms.mpi import MainMPIComm

    if not libE_specs["disable_log_files"]:
        exit_logger = manager_logging_config(specs=libE_specs)
    else:
        exit_logger = None

    if libE_specs.get("nworkers"):
        logger.manager_warning("*WARNING* 'mpi' comms is detected with 'nworkers'. nworkers will be ignored.\n")

    if isinstance(Executor.executor, MPIExecutor):
        if Executor.executor.mpi_runner_type == "openmpi":
            logger.manager_warning(
                "WARNING: Nested MPI-workflow detected. User initialized both an MPI runtime and an MPI Executor.\n"
                + "       Expect complications if using Open MPI."
                + " An MPICH-derived MPI distribution is recommended for nested MPI workflows. \n"
            )

    def cleanup():
        """Process cleanup required on exit"""
        if exit_logger is not None:
            exit_logger()

    # Set up abort handler
    def on_abort():
        """Shut down MPI on error."""
        comms_abort(mpi_comm)

    # Run generic manager
    return manager(
        [MainMPIComm(mpi_comm, w) for w in range(1, mpi_comm.Get_size())],
        sim_specs,
        gen_specs,
        exit_criteria,
        persis_info,
        alloc_specs,
        libE_specs,
        History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0),
        on_abort=on_abort,
        on_cleanup=cleanup,
    )


def libE_mpi_worker(libE_comm, sim_specs, gen_specs, libE_specs):
    """Worker routines run on ranks > 0."""
    from libensemble.comms.mpi import MainMPIComm

    comm = MainMPIComm(libE_comm)
    worker_main(comm, sim_specs, gen_specs, libE_specs, log_comm=True)
    logger.debug(f"Worker {libE_comm.Get_rank()} exiting")
    return [], {}, []


# ==================== Local version ===============================


def start_proc_team(nworkers, sim_specs, gen_specs, libE_specs, log_comm=True):
    """Launch a process worker team."""
    resources = Resources.resources
    executor = Executor.executor

    if libE_specs["comms"] == "local":
        QCommLocal = QCommProcess
    else:  # threads
        QCommLocal = QCommThread
        log_comm = False  # Prevents infinite loop of logging.

    wcomms = [
        QCommLocal(worker_main, nworkers, sim_specs, gen_specs, libE_specs, w, log_comm, resources, executor)
        for w in range(1, nworkers + 1)
    ]

    for wcomm in wcomms:
        wcomm.run()

    return wcomms


def kill_proc_team(wcomms, timeout):
    """Join on workers (and terminate forcefully if needed)."""
    for wcomm in wcomms:
        try:
            wcomm.result(timeout=timeout)
        except Timeout:
            wcomm.terminate()


def libE_local(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0):
    """Main routine for thread/process launch of libE."""

    resources = Resources.resources
    if resources is not None:
        local_host = [socket.gethostname()]
        resources.add_comm_info(libE_nodes=local_host)

    exctr = Executor.executor
    if exctr is not None:
        exctr.set_resources(resources)
        exctr.serial_setup()

    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Launch worker team and set up logger
    wcomms = start_proc_team(libE_specs["nworkers"], sim_specs, gen_specs, libE_specs)

    # Set manager resources after the forkpoint.
    if resources is not None:
        resources.set_resource_manager(libE_specs["nworkers"])

    if not libE_specs["disable_log_files"]:
        exit_logger = manager_logging_config(specs=libE_specs)
    else:
        exit_logger = None

    # Set up cleanup routine to shut down worker team
    def cleanup():
        """Handler to clean up comms team."""
        kill_proc_team(wcomms, timeout=libE_specs["worker_timeout"])
        if exit_logger is not None:
            exit_logger()

    # Run generic manager
    return manager(
        wcomms, sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, hist, on_cleanup=cleanup
    )


# ==================== TCP version =================================


def get_ip():
    """Get the IP address of the current host"""
    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        return "localhost"


def libE_tcp_default_ID():
    """Assign a (we hope unique) worker ID if not assigned by manager."""
    return f"{get_ip()}_pid{os.getpid()}"


def libE_tcp(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0):
    """Main routine for TCP multiprocessing launch of libE."""

    is_worker = libE_specs.get("workerID") is not None

    exctr = Executor.executor
    if exctr is not None:
        # TCP does not currently support resource_management but when does, assume
        # each TCP worker is in a different resource pool (only knowing local_host)
        if not is_worker:
            exctr.serial_setup()

    if is_worker:
        libE_tcp_worker(sim_specs, gen_specs, libE_specs)
        return [], persis_info, []

    return libE_tcp_mgr(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0)


def libE_tcp_worker_launcher(libE_specs):
    """Get a launch function from libE_specs."""
    if "worker_launcher" in libE_specs:
        worker_launcher = libE_specs["worker_launcher"]
    else:
        worker_cmd = libE_specs["worker_cmd"]

        def worker_launcher(specs):
            """Basic worker launch function."""
            return launcher.launch(worker_cmd, specs)

    return worker_launcher


def libE_tcp_start_team(manager, nworkers, workers, ip, port, authkey, launchf):
    """Launch nworkers workers that attach back to a managers server."""
    worker_procs = []
    specs = {"manager_ip": ip, "manager_port": port, "authkey": authkey}
    with Timer() as timer:
        for w in range(1, nworkers + 1):
            logger.info(f"Manager is launching worker {w}")
            if workers is not None:
                specs["worker_ip"] = workers[w - 1]
                specs["tunnel_port"] = 0x71BE
            specs["workerID"] = w
            worker_procs.append(launchf(specs))
        logger.info(f"Manager is awaiting {nworkers} workers")
        wcomms = manager.await_workers(nworkers)
        logger.info(f"Manager connected to {nworkers} workers ({timer.elapsed} s)")
    return worker_procs, wcomms


def libE_tcp_mgr(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0):
    """Main routine for TCP multiprocessing launch of libE at manager."""
    hist = History(alloc_specs, sim_specs, gen_specs, exit_criteria, H0)

    # Set up a worker launcher
    launchf = libE_tcp_worker_launcher(libE_specs)

    # Get worker launch parameters and fill in defaults for TCP/IP conn
    if libE_specs.get("nworkers"):
        workers = None
        nworkers = libE_specs["nworkers"]
    elif libE_specs.get("workers"):
        workers = libE_specs["workers"]
        nworkers = len(workers)
    ip = libE_specs["ip"] or get_ip()
    port = libE_specs["port"]
    authkey = libE_specs["authkey"]

    with ServerQCommManager(port, authkey.encode("utf-8")) as tcp_manager:
        # Get port if needed because of auto-assignment
        if port == 0:
            _, port = tcp_manager.address

        if not libE_specs["disable_log_files"]:
            exit_logger = manager_logging_config(specs=libE_specs)
        else:
            exit_logger = None

        logger.info(f"Launched server at ({ip}, {port})")

        # Launch worker team and set up logger
        worker_procs, wcomms = libE_tcp_start_team(tcp_manager, nworkers, workers, ip, port, authkey, launchf)

        def cleanup():
            """Handler to clean up launched team."""
            for wp in worker_procs:
                launcher.cancel(wp, timeout=libE_specs["worker_timeout"])
            if exit_logger is not None:
                exit_logger()

        # Run generic manager
        return manager(
            wcomms, sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, hist, on_cleanup=cleanup
        )


def libE_tcp_worker(sim_specs, gen_specs, libE_specs):
    """Main routine for TCP worker launched by libE."""
    ip = libE_specs["ip"]
    port = libE_specs["port"]
    authkey = libE_specs["authkey"]
    workerID = libE_specs["workerID"]

    with ClientQCommManager(ip, port, authkey.encode("utf-8"), workerID) as comm:
        worker_main(comm, sim_specs, gen_specs, libE_specs, workerID=workerID, log_comm=True)
        logger.debug(f"Worker {workerID} exiting")


# ==================== Additional Internal Functions ===========================


def _dump_on_abort(hist, persis_info, save_H=True, path=Path.cwd()):
    """Dump history and persis_info on abort"""
    logger.error("Manager exception raised .. aborting ensemble:")
    logger.error(f"Dumping ensemble history with {hist.sim_ended_count} sims evaluated:")

    if save_H:
        np.save(Path(path / Path("libE_history_at_abort_" + str(hist.sim_ended_count) + ".npy")), hist.trim_H())
        with Path(path / Path("libE_persis_info_at_abort_" + str(hist.sim_ended_count) + ".pickle")).open("wb") as f:
            pickle.dump(persis_info, f)
