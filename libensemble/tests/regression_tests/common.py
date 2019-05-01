"""
Common plumbing for regression tests
"""

import sys
import os
import os.path
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser(prog='test_...')
parser.add_argument('--comms', type=str, nargs='?',
                    choices=['local', 'tcp', 'ssh', 'client', 'mpi'],
                    default='mpi', help='Type of communicator')
parser.add_argument('--nworkers', type=int, nargs='?',
                    help='Number of local forked processes')
parser.add_argument('--workers', type=str, nargs='+',
                    help='List of worker nodes')
parser.add_argument('--workerID', type=int, nargs='?', help='Client worker ID')
parser.add_argument('--server', type=str, nargs=3,
                    help='Triple of (ip, port, authkey) used to reach manager')
parser.add_argument('--pwd', type=str, nargs='?',
                    help='Working directory to be used')
parser.add_argument('--worker_pwd', type=str, nargs='?',
                    help='Working directory on remote client')
parser.add_argument('--worker_python', type=str, nargs='?',
                    default=sys.executable,
                    help='Python version on remote client')
parser.add_argument('--tester_args', type=str, nargs='*',
                    help='Additional arguments for use by specific testers')


def mpi_parse_args(args):
    "Parse arguments for MPI comms."
    from mpi4py import MPI
    nworkers = MPI.COMM_WORLD.Get_size()-1
    is_master = MPI.COMM_WORLD.Get_rank() == 0
    libE_specs = {'comm': MPI.COMM_WORLD, 'color': 0, 'comms': 'mpi'}
    return nworkers, is_master, libE_specs, args.tester_args


def local_parse_args(args):
    "Parse arguments for forked processes using multiprocessing."
    nworkers = args.nworkers or 4
    libE_specs = {'nprocesses': nworkers, 'comms': 'local'}
    return nworkers, True, libE_specs, args.tester_args


def tcp_parse_args(args):
    "Parse arguments for local TCP connections"
    nworkers = args.nworkers or 4
    cmd = [
        sys.executable, sys.argv[0], "--comms", "client", "--server",
        "{manager_ip}", "{manager_port}", "{authkey}", "--workerID",
        "{workerID}", "--nworkers",
        str(nworkers)]
    libE_specs = {'nprocesses': nworkers, 'worker_cmd': cmd, 'comms': 'tcp'}
    return nworkers, True, libE_specs, args.tester_args


def ssh_parse_args(args):
    "Parse arguments for SSH with reverse tunnel."
    nworkers = len(args.workers)
    worker_pwd = args.worker_pwd or os.getcwd()
    script_dir, script_name = os.path.split(sys.argv[0])
    worker_script_name = os.path.join(worker_pwd, script_name)
    ssh = [
        "ssh", "-R", "{tunnel_port}:localhost:{manager_port}", "{worker_ip}"]
    cmd = [
        args.worker_python, worker_script_name, "--comms", "client",
        "--server", "localhost", "{tunnel_port}", "{authkey}", "--workerID",
        "{workerID}", "--nworkers",
        str(nworkers)]
    cmd = " ".join(cmd)
    cmd = "( cd {} ; {} )".format(worker_pwd, cmd)
    ssh.append(cmd)
    libE_specs = {'workers': args.workers,
                  'worker_cmd': ssh,
                  'ip': 'localhost',
                  'comms': 'tcp'}
    return nworkers, True, libE_specs, args.tester_args


def client_parse_args(args):
    "Parse arguments for a TCP client."
    nworkers = args.nworkers or 4
    ip, port, authkey = args.server
    libE_specs = {'ip': ip,
                  'port': int(port),
                  'authkey': authkey.encode('utf-8'),
                  'workerID': args.workerID,
                  'nprocesses': nworkers,
                  'comms': 'tcp'}
    return nworkers, False, libE_specs, args.tester_args


def parse_args():
    "Unified parsing interface for regression test arguments"
    args = parser.parse_args(sys.argv[1:])
    front_ends = {
        'mpi': mpi_parse_args,
        'local': local_parse_args,
        'tcp': tcp_parse_args,
        'ssh': ssh_parse_args,
        'client': client_parse_args}
    if args.pwd is not None:
        os.chdir(args.pwd)
    return front_ends[args.comms or 'mpi'](args)


def save_libE_output(H, persis_info, calling_file, nworkers):
    script_name = os.path.splitext(os.path.basename(calling_file))[0]
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_History_length=' + str(len(H)) \
                          + '_evals=' + str(sum(H['returned'])) \
                          + '_ranks=' + str(nworkers)

    print("\n\n\nRun completed.\nSaving results to file: "+filename)
    np.save(filename, H)

    with open(filename + ".pickle", "wb") as f:
        pickle.dump(persis_info, f)


def per_worker_stream(persis_info, nworkers):
    for i in range(nworkers):
        if i in persis_info:
            persis_info[i].update({
                'rand_stream': np.random.RandomState(i),
                'worker_num': i})
        else:
            persis_info[i] = {
                'rand_stream': np.random.RandomState(i),
                'worker_num': i}
    return persis_info


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def build_simfunc():
    import subprocess

    # Build simfunc
    # buildstring='mpif90 -o my_simjob.x my_simjob.f90' # On cray need to use ftn
    buildstring = 'mpicc -o my_simjob.x ../unit_tests/simdir/my_simjob.c'
    # subprocess.run(buildstring.split(),check=True) #Python3.5+
    subprocess.check_call(buildstring.split())
