"""
Common plumbing for regression tests
"""

import os
import os.path


def mpi_comm_excl(exc=[0], comm=None):
    "Exlude ranks from a communicator for MPI comms."
    from mpi4py import MPI
    parent_comm = comm or MPI.COMM_WORLD
    parent_group = parent_comm.Get_group()
    new_group = parent_group.Excl(exc)
    mpi_comm = parent_comm.Create(new_group)
    return mpi_comm, MPI.COMM_NULL


def mpi_comm_split(num_parts, comm=None):
    "Split COMM_WORLD into sub-communicators for MPI comms."
    from mpi4py import MPI
    parent_comm = comm or MPI.COMM_WORLD
    parent_size = parent_comm.Get_size()
    key = parent_comm.Get_rank()
    row_size = parent_size // num_parts
    sub_comm_number = key // row_size
    sub_comm = parent_comm.Split(sub_comm_number, key)
    return sub_comm, sub_comm_number


def build_simfunc():
    import subprocess

    # Build simfunc
    # buildstring='mpif90 -o my_simjob.x my_simjob.f90' # On cray need to use ftn
    buildstring = 'mpicc -o my_simjob.x ../unit_tests/simdir/my_simjob.c'
    # subprocess.run(buildstring.split(),check=True) #Python3.5+
    subprocess.check_call(buildstring.split())


def modify_Balsam_worker():
    # Balsam is meant for HPC systems that commonly distribute jobs across many
    #   nodes. Due to the nature of testing Balsam on local or CI systems which
    #   usually only contain a single node, we need to change Balsam's default
    #   worker setup so multiple workers can be run on a single node.
    import balsam

    new_lines = ["        for idx in range(10):\n",
                 "            w = Worker(1, host_type='DEFAULT', num_nodes=1)\n",
                 "            self.workers.append(w)\n"]

    workerfile = 'worker.py'
    balsam_path = os.path.dirname(balsam.__file__) + '/launcher'
    balsam_worker_path = os.path.join(balsam_path, workerfile)

    with open(balsam_worker_path, 'r') as f:
        lines = f.readlines()

    if lines[-3] != new_lines[0]:
        lines = lines[:-2]  # effectively inserting new_lines[0] above
        lines.extend(new_lines)

    with open(balsam_worker_path, 'w') as f:
        for line in lines:
            f.write(line)


def modify_Balsam_pyCoverage():
    # Tracking line coverage through our tests requires running the Python module
    #   'coverage' directly. Balsam explicitely configures Python runs with
    #   'python [script].py args' with no current capability for specifying
    #   modules. This hack specifies the coverage module and some options.
    import balsam

    old_line = "            path = ' '.join((exe, script_path, args))\n"
    new_line = "            path = ' '.join((exe, '-m coverage run " + \
               "--parallel-mode --rcfile=./libensemble/tests/regression_tests/" + \
               ".bal_coveragerc', script_path, args))\n"

    commandfile = 'cli_commands.py'
    balsam_path = os.path.dirname(balsam.__file__) + '/scripts'
    balsam_commands_path = os.path.join(balsam_path, commandfile)

    with open(balsam_commands_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if lines[i] == old_line:
            lines[i] = new_line

    with open(balsam_commands_path, 'w') as f:
        for line in lines:
            f.write(line)


def modify_Balsam_JobEnv():
    # If Balsam detects that the system on which it is running contains the string
    #   'cc' in it's hostname, then it thinks it's on Cooley! Travis hostnames are
    #   randomly generated and occasionally may contain that offending string. This
    #   modifies Balsam's JobEnvironment class to not check for 'cc'.
    import balsam

    bad_line = "        'COOLEY' : 'cooley cc'.split()\n"
    new_line = "        'COOLEY' : 'cooley'.split()\n"

    jobenv_file = 'JobEnvironment.py'
    balsam_path = os.path.dirname(balsam.__file__) + '/service/schedulers'
    balsam_jobenv_path = os.path.join(balsam_path, jobenv_file)

    with open(balsam_jobenv_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if lines[i] == bad_line:
            lines[i] = new_line

    with open(balsam_jobenv_path, 'w') as f:
        for line in lines:
            f.write(line)
