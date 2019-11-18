"""
libEnsemble utilities
============================================

"""

__all__ = ['check_inputs']

import os
import sys
import logging
import numpy as np
import argparse
import pickle

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


"""
Below are the fields used within libEnsemble
"""
libE_fields = [('sim_id', int),        # Unique id of entry in H that was generated
               ('gen_worker', int),    # Worker that generated the entry
               ('gen_time', float),    # Time (since epoch) entry was entered into H
               ('given', bool),        # True if entry has been given for sim eval
               ('returned', bool),     # True if entry has been returned from sim eval
               ('given_time', float),  # Time (since epoch) that the entry was given
               ('sim_worker', int),    # Worker that did (or is doing) the sim eval
               ]
# end_libE_fields_rst_tag

allowed_sim_spec_keys = ['sim_f',  #
                         'in',     #
                         'out',    #
                         'user']   #

allowed_gen_spec_keys = ['gen_f',  #
                         'in',     #
                         'out',    #
                         'user']   #

allowed_alloc_spec_keys = ['alloc_f',  #
                           'in',       #
                           'out',      #
                           'user']     #

allowed_libE_spec_keys = ['comms',               #
                          'comm',                #
                          'ip',                  #
                          'port',                #
                          'authkey',             #
                          'workerID',            #
                          'nworkers',          #
                          'worker_cmd',          #
                          'abort_on_exception',  #
                          'sim_dir',             #
                          'sim_dir_prefix',      #
                          'sim_dir_suffix',      #
                          'clean_jobs',          #
                          'save_every_k_sims',   #
                          'save_every_k_gens',   #
                          'profile_worker']      #

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


def check_libE_specs(libE_specs, serial_check=False):
    assert isinstance(libE_specs, dict), "libE_specs must be a dictionary"
    comms_type = libE_specs.get('comms', 'mpi')
    if comms_type in ['mpi']:
        if not serial_check:
            assert libE_specs['comm'].Get_size() > 1, "Manager only - must be at least one worker (2 MPI tasks)"
    elif comms_type in ['local']:
        assert libE_specs['nworkers'] >= 1, "Must specify at least one worker"
    elif comms_type in ['tcp']:
        # TODO, differentiate and test SSH/Client
        assert libE_specs['nworkers'] >= 1, "Must specify at least one worker"

    for k in libE_specs.keys():
        assert k in allowed_libE_spec_keys, "Key %s is not allowed in libE_specs. Supported keys are: %s " % (k, allowed_libE_spec_keys)


def check_alloc_specs(alloc_specs):
    assert isinstance(alloc_specs, dict), "alloc_specs must be a dictionary"
    assert alloc_specs['alloc_f'], "Allocation function must be specified"

    for k in alloc_specs.keys():
        assert k in allowed_alloc_spec_keys, "Key %s is not allowed in alloc_specs. Supported keys are: %s " % (k, allowed_alloc_spec_keys)


def check_sim_specs(sim_specs):
    assert isinstance(sim_specs, dict), "sim_specs must be a dictionary"
    assert any([term_field in sim_specs for term_field in ['sim_f', 'in', 'out']]), \
        "sim_specs must contain 'sim_f', 'in', 'out'"

    assert len(sim_specs['out']), "sim_specs must have 'out' entries"
    assert isinstance(sim_specs['in'], list), "'in' field must exist and be a list of field names"

    for k in sim_specs.keys():
        assert k in allowed_sim_spec_keys, "Key %s is not allowed in sim_specs. Supported keys are: %s " % (k, allowed_sim_spec_keys)


def check_gen_specs(gen_specs):
    assert isinstance(gen_specs, dict), "gen_specs must be a dictionary"
    assert not bool(gen_specs) or len(gen_specs['out']), "gen_specs must have 'out' entries"

    for k in gen_specs.keys():
        assert k in allowed_gen_spec_keys, "Key %s is not allowed in gen_specs. Supported keys are: %s " % (k, allowed_gen_spec_keys)


def check_exit_criteria(exit_criteria, sim_specs, gen_specs):
    assert isinstance(exit_criteria, dict), "exit_criteria must be a dictionary"

    assert len(exit_criteria) > 0, "Must have some exit criterion"

    # Ensure termination criteria are valid
    valid_term_fields = ['sim_max', 'gen_max',
                         'elapsed_wallclock_time', 'stop_val']
    assert all([term_field in valid_term_fields for term_field in exit_criteria]), \
        "Valid termination options: " + str(valid_term_fields)

    # Make sure stop-values match parameters in gen_specs or sim_specs
    if 'stop_val' in exit_criteria:
        stop_name = exit_criteria['stop_val'][0]
        sim_out_names = [e[0] for e in sim_specs['out']]
        gen_out_names = [e[0] for e in gen_specs['out']]
        assert stop_name in sim_out_names + gen_out_names, \
            "Can't stop on {} if it's not in a sim/gen output".format(stop_name)


def check_H(H0, sim_specs, alloc_specs, gen_specs):
    if len(H0):
        # Set up dummy history to see if it agrees with H0
        Dummy_H = np.zeros(1 + len(H0), dtype=libE_fields + list(set(sum([k['out'] for k in [sim_specs, alloc_specs, gen_specs] if k], []))))  # Combines all 'out' fields (if they exist) in sim_specs, gen_specs, or alloc_specs

        fields = H0.dtype.names

        # Prior history must contain the fields in new history
        assert set(fields).issubset(set(Dummy_H.dtype.names)), \
            "H0 contains fields {} not in the History.".\
            format(set(fields).difference(set(Dummy_H.dtype.names)))

        # Prior history cannot contain unreturned points
        # assert 'returned' not in fields or np.all(H0['returned']), \
        #     "H0 contains unreturned points."

        # Fail if prior history contains unreturned points (or returned but not given).
        assert('returned' not in fields or np.all(H0['given'] == H0['returned'])), \
            'H0 contains unreturned or invalid points'

        # Check dimensional compatibility of fields
        for field in fields:
            check_consistent_field(field, H0[field], Dummy_H[field])


def check_inputs(libE_specs=None, alloc_specs=None, sim_specs=None, gen_specs=None, exit_criteria=None, H0=None, serial_check=False):
    """
    Check if the libEnsemble arguments are of the correct data type and contain
    sufficient information to perform a run. There is no return value. An
    exception is raised if any of the checks fail.

    Parameters
    ----------

    libE_specs, alloc_specs, sim_specs, gen_specs, exit_criteria: :obj:`dict`, optional

        libEnsemble data structures

    H0: :obj:`numpy structured array`, optional

        A previous libEnsemble history to be prepended to the history in the
        current libEnsemble run
        :doc:`(example)<data_structures/history_array>`

    serial_check : :obj:`boolean`

        If True, assumes running a serial check. This means, for example,
        the details of current MPI communicator are not checked (can be
        run with libE_specs{'comm': 'mpi'} without running through mpiexec.

    """
    # Detailed checking based on Required Keys in docs for each specs
    if libE_specs is not None:
        check_libE_specs(libE_specs, serial_check)

    if alloc_specs is not None:
        check_alloc_specs(alloc_specs)

    if sim_specs is not None:
        check_sim_specs(sim_specs)

    if gen_specs is not None:
        check_gen_specs(gen_specs)

    if exit_criteria is not None:
        assert sim_specs is not None and gen_specs is not None, \
            "Can't check exit_criteria without sim_specs and gen_specs"
        check_exit_criteria(exit_criteria, sim_specs, gen_specs)

    if H0 is not None:
        assert sim_specs is not None and alloc_specs is not None and gen_specs is not None, \
            "Can't check H0 without sim_specs, alloc_specs, gen_specs"
        check_H(H0, sim_specs, alloc_specs, gen_specs)

# ==================== Command-line argument parsing ===========================


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
    libE_specs = {'comm': MPI.COMM_WORLD, 'comms': 'mpi'}
    return nworkers, is_master, libE_specs, args.tester_args


def local_parse_args(args):
    "Parse arguments for forked processes using multiprocessing."
    nworkers = args.nworkers or 4
    libE_specs = {'nworkers': nworkers, 'comms': 'local'}
    return nworkers, True, libE_specs, args.tester_args


def tcp_parse_args(args):
    "Parse arguments for local TCP connections"
    nworkers = args.nworkers or 4
    cmd = [
        sys.executable, sys.argv[0], "--comms", "client", "--server",
        "{manager_ip}", "{manager_port}", "{authkey}", "--workerID",
        "{workerID}", "--nworkers",
        str(nworkers)]
    libE_specs = {'nworkers': nworkers, 'worker_cmd': cmd, 'comms': 'tcp'}
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
                  'nworkers': nworkers,
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

# =================== save libE output to pickle and np ========================


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

# ===================== per-worker numpy random-streams ========================


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
