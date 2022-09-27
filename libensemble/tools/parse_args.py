import sys
import os
import argparse

from libensemble.tools.tools import logger

# ==================== Command-line argument parsing ===========================

parser = argparse.ArgumentParser(prog="test_...")

parser.add_argument(
    "--comms",
    type=str,
    nargs="?",
    choices=["local", "tcp", "ssh", "client", "mpi"],
    default="mpi",
    help="Type of communicator",
)
parser.add_argument("--nworkers", type=int, nargs="?", help="Number of local forked processes")
parser.add_argument(
    "--nsim_workers",
    type=int,
    nargs="?",
    help="Number of workers for sims. 1+ unresourced workers for a persistent generator will be added",
)
parser.add_argument("--nresource_sets", type=int, nargs="?", help="Number of resource sets")
parser.add_argument("--workers", type=str, nargs="+", help="List of worker nodes")
parser.add_argument("--workerID", type=int, nargs="?", help="Client worker ID")
parser.add_argument("--server", type=str, nargs=3, help="Triple of (ip, port, authkey) used to reach manager")
parser.add_argument("--pwd", type=str, nargs="?", help="Working directory to be used")
parser.add_argument("--worker_pwd", type=str, nargs="?", help="Working directory on remote client")
parser.add_argument(
    "--worker_python", type=str, nargs="?", default=sys.executable, help="Python version on remote client"
)
parser.add_argument("--tester_args", type=str, nargs="*", help="Additional arguments for use by specific testers")


def _get_zrw(nworkers, nsim_workers):
    """Determine zero resource workers from workers and sim workers"""
    ngen_workers = nworkers - nsim_workers
    assert ngen_workers > 0, "nsim_workers cannot be greater than number of workers"
    return [i for i in range(1, ngen_workers + 1)]


def _mpi_parse_args(args):
    """Parses arguments for MPI comms."""
    from mpi4py import MPI, rc

    if rc.initialize is False and not MPI.Is_initialized():
        MPI.Init()

    nworkers = MPI.COMM_WORLD.Get_size() - 1
    is_manager = MPI.COMM_WORLD.Get_rank() == 0
    libE_specs = {"mpi_comm": MPI.COMM_WORLD, "comms": "mpi"}

    if args.nresource_sets is not None:
        libE_specs["num_resource_sets"] = args.nresource_sets

    # Convenience option which sets other libE_specs options.
    nsim_workers = args.nsim_workers
    if nsim_workers is not None:
        # libE_specs["zero_resource_workers"] = _get_zrw(nworkers, nsim_workers)
        libE_specs["num_resource_sets"] = libE_specs.get("num_resource_sets", nsim_workers)

    return nworkers, is_manager, libE_specs, args.tester_args


def _local_parse_args(args):
    """Parses arguments for forked processes using multiprocessing."""
    libE_specs = {"comms": "local"}
    nworkers = args.nworkers

    if args.nresource_sets is not None:
        libE_specs["num_resource_sets"] = args.nresource_sets

    # Convenience option which sets other libE_specs options.
    nsim_workers = args.nsim_workers
    if nsim_workers is not None:
        nworkers = nworkers or nsim_workers + 1
        # libE_specs["zero_resource_workers"] = _get_zrw(nworkers, nsim_workers)
        libE_specs["num_resource_sets"] = libE_specs.get("num_resource_sets", nsim_workers)

    nworkers = nworkers or 4
    libE_specs["nworkers"] = nworkers

    return nworkers, True, libE_specs, args.tester_args


def _tcp_parse_args(args):
    """Parses arguments for local TCP connections"""
    nworkers = args.nworkers or 4
    cmd = [
        sys.executable,
        sys.argv[0],
        "--comms",
        "client",
        "--server",
        "{manager_ip}",
        "{manager_port}",
        "{authkey}",
        "--workerID",
        "{workerID}",
        "--nworkers",
        str(nworkers),
    ]
    libE_specs = {"nworkers": nworkers, "worker_cmd": cmd, "comms": "tcp"}
    return nworkers, True, libE_specs, args.tester_args


def _ssh_parse_args(args):
    """Parses arguments for SSH with reverse tunnel."""
    nworkers = len(args.workers)
    worker_pwd = args.worker_pwd or os.getcwd()
    script_dir, script_name = os.path.split(sys.argv[0])
    worker_script_name = os.path.join(worker_pwd, script_name)
    ssh = ["ssh", "-R", "{tunnel_port}:localhost:{manager_port}", "{worker_ip}"]
    cmd = [
        args.worker_python,
        worker_script_name,
        "--comms",
        "client",
        "--server",
        "localhost",
        "{tunnel_port}",
        "{authkey}",
        "--workerID",
        "{workerID}",
        "--nworkers",
        str(nworkers),
    ]
    cmd = " ".join(cmd)
    cmd = f"( cd {worker_pwd} ; {cmd} )"
    ssh.append(cmd)
    libE_specs = {"workers": args.workers, "worker_cmd": ssh, "ip": "localhost", "comms": "tcp"}
    return nworkers, True, libE_specs, args.tester_args


def _client_parse_args(args):
    """Parses arguments for a TCP client."""
    nworkers = args.nworkers or 4
    ip, port, authkey = args.server
    libE_specs = {
        "ip": ip,
        "port": int(port),
        "authkey": authkey.encode("utf-8"),
        "workerID": args.workerID,
        "nworkers": nworkers,
        "comms": "tcp",
    }
    return nworkers, False, libE_specs, args.tester_args


def parse_args():
    """
    Parses command-line arguments. Use in calling script.

    .. code-block:: python

        from libensemble.tools import parse_args
        nworkers, is_manager, libE_specs, misc_args = parse_args()

    From the shell::

        $ python calling_script --comms local --nworkers 4

    Usage:

    .. code-block:: bash

        usage: test_... [-h] [--comms [{local, tcp, ssh, client, mpi}]]
                        [--nworkers [NWORKERS]] [--workers WORKERS [WORKERS ...]]
                        [--nsim_workers [NSIM_WORKERS]]
                        [--nresource_sets [NRESOURCE_SETS]]
                        [--workerID [WORKERID]] [--server SERVER SERVER SERVER]
                        [--pwd [PWD]] [--worker_pwd [WORKER_PWD]]
                        [--worker_python [WORKER_PYTHON]]
                        [--tester_args [TESTER_ARGS [TESTER_ARGS ...]]]

        Note that running via an MPI runner uses the default 'mpi' comms, and '--nworkers'
        will be ignored. The number of processes are supplied via the MPI run line. One being
        the manager, and the rest are workers.

        --comms,          Communications medium for manager and workers. Default is 'mpi'.
        --nworkers,       (For 'local' or 'tcp' comms) Set number of workers.
        --nresource_sets, Explicitly set the number of resource sets. This sets
                          libE_specs['num_resource_sets']. By default, resources will be
                          divided by workers (excluding zero_resource_workers).
        --nsim_workers,   (For 'local' or 'mpi' comms) A convenience option for cases with
                          persistent generators - sets the number of simulation workers.
                          If used with no other criteria, one additional worker for running a
                          generator will be added, and the number of resource sets will be assigned
                          the given value. If '--nworkers' has also been specified, will generate
                          enough additional workers to match the other criteria. If '--nresource_sets'
                          is also specified, will not override resource sets.

        Example command lines:

        Run with 'local' comms and 4 workers
        $ python calling_script --comms local --nworkers 4

        Run with 'local' comms and 5 workers - one gen worker (no resources), and 4 sim workers.
        $ python calling_script --comms local --nsim_workers 4

        Run with 'local' comms with 4 workers and 8 resource sets. The extra resource sets will
        be used for larger simulations (using variable resource assignment).
        $ python calling_script --comms local --nresource_sets 8

        Previous example with 'mpi' comms.
        $ mpirun -np 5 python calling_script --nresource_sets 8

    Returns
    -------

    nworkers: :obj:`int`
        Number of workers libEnsemble will initiate

    is_manager: :obj:`boolean`
        Indicates whether the current process is the manager process

    libE_specs: :obj:`dict`
        Settings and specifications for libEnsemble
        :doc:`(example)<data_structures/libE_specs>`

    """
    args, unknown = parser.parse_known_args(sys.argv[1:])
    front_ends = {
        "mpi": _mpi_parse_args,
        "local": _local_parse_args,
        "tcp": _tcp_parse_args,
        "ssh": _ssh_parse_args,
        "client": _client_parse_args,
    }
    if args.pwd is not None:
        os.chdir(args.pwd)
    nworkers, is_manager, libE_specs, tester_args = front_ends[args.comms or "mpi"](args)
    if is_manager and unknown:
        logger.warning(f"parse_args ignoring unrecognized arguments: {' '.join(unknown)}")
    return nworkers, is_manager, libE_specs, tester_args
