"""
This module contains various versions that evaluate the six hump camel function.
"""
__all__ = ['six_hump_camel_CUDA_variable_resources', 'six_hump_camel_with_different_resources',
           'six_hump_camel', 'six_hump_camel_simple']

# import subprocess
import os
import sys
import numpy as np
import time
from libensemble.executors.executor import Executor
from libensemble.message_numbers import UNSET_TAG, WORKER_DONE, TASK_FAILED
from libensemble.resources.resources import Resources


# SH TODO: Resource variable names are subject to change
def six_hump_camel_CUDA_variable_resources(H, persis_info, sim_specs, libE_info):
    """Launches an app setting GPU resources

    The standard test apps do not run on GPU, but demonstrates accessing resource
    information to set CUDA_VISIBLE_DEVICES, and typical run configuration.
    """
    x = H['x'][0]
    H_o = np.zeros(1, dtype=sim_specs['out'])

    # Interrogate resources available to this worker
    resources = Resources.resources.worker_resources
    if resources.even_slots:  # Need same slots on each node
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # To order by PCI  bus IDs
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, resources.slots_on_node))  # List to string
        num_nodes = resources.local_node_count
        cores_per_node = resources.slot_count  # One CPU per GPU
        #print('CUDA_VISIBLE_DEVICES is {} nodes {} ppn {}'
              #.format(os.environ["CUDA_VISIBLE_DEVICES"], num_nodes, cores_per_node), flush=True)
    else:
        # Unless use a matching sub-set, but usually you probably don't want this
        print('Error: Cannot set CUDA_VISIBLE_DEVICES when uneven slots on nodes {}'.format(resources.slots))

    # Create application input file
    inpt = ' '.join(map(str, x))
    exctr = Executor.executor  # Get Executor

    # Launch application via system MPI runner, using assigned resources.
    task = exctr.submit(app_name='six_hump_camel',
                        app_args=inpt,
                        num_nodes=num_nodes,
                        ranks_per_node=cores_per_node,
                        stdout='out.txt',
                        stderr='err.txt')

    task.wait()  # Wait for run to complete

    # Access app output
    with open('out.txt') as f:
        H_o['f'] = float(f.readline().strip())  # Read just first line

    calc_status = WORKER_DONE if task.state == 'FINISHED' else 'FAILED'
    return H_o, persis_info, calc_status


# SH TODO: Should we move this to below
#          Check/update docstring
def six_hump_camel_with_different_resources(H, persis_info, sim_specs, libE_info):
    """
    Evaluates the six hump camel for a collection of points given in ``H['x']`` but also
    performs a system call with a given number of nodes and ranks per node
    using a machinefile (to show one way of evaluating a compiled simulation).

    .. seealso::
        `test_uniform_sampling_with_different_resources.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling_with_different_resources.py>`_ # noqa
    """

    batch = len(H['x'])
    H_o = np.zeros(batch, dtype=sim_specs['out'])
    app = sim_specs['user'].get('app', 'helloworld')
    dry_run = sim_specs['user'].get('dry_run', False)  # dry_run only prints run lines in ensemble.log
    core_multiplier = 1  # Only used if resource_sets is passed in H['in']

    exctr = Executor.executor  # Get Executor
    task_states = []
    for i, x in enumerate(H['x']):
        # If passing resource sets in, use that here (you can oversubscribe on node)
        # else let automatic resources set nodes/procs to use all available set.
        nprocs = None  # Will be as if argument is not present
        if 'resource_sets' in sim_specs['in']:
            nprocs = H['resource_sets'][i] * core_multiplier
            # print('nprocs is',nprocs,flush=True)

        inpt = None  # Will be as if argument is not present
        if app == 'six_hump_camel':
            inpt = ' '.join(map(str, H['x'][i]))

        task = exctr.submit(app_name=app, app_args=inpt,
                            num_procs=nprocs,
                            stdout='out.txt', stderr='err.txt',
                            dry_run=dry_run)
        task.wait()
        # while(not task.finished):
        #     time.sleep(0.1)
        #     task.poll()

        task_states.append(task.state)

        if app == 'six_hump_camel':
            # H_o['f'][i] = float(task.read_stdout())  # Reads whole file
            with open('out.txt') as f:
                H_o['f'][i] = float(f.readline().strip())  # Read just first line
        else:
            # To return something in test
            H_o['f'][i] = six_hump_camel_func(x)

        # v = np.random.uniform(0, 10)
        # print('About to sleep for :' + str(v))
        # time.sleep(v)

    calc_status = UNSET_TAG  # Returns to worker
    if all(t == 'FINISHED' for t in task_states):
        calc_status = WORKER_DONE
    elif any(t == 'FAILED' for t in task_states):
        calc_status = TASK_FAILED

    return H_o, persis_info, calc_status


def six_hump_camel(H, persis_info, sim_specs, _):
    """
    Evaluates the six hump camel function for a collection of points given in ``H['x']``.
    Additionally evaluates the gradient if ``'grad'`` is a field in
    ``sim_specs['out']`` and pauses for ``sim_specs['user']['pause_time']]`` if
    defined.

    .. seealso::
        `test_old_aposmm_with_gradients.py  <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_old_aposmm_with_gradients.py>`_ # noqa
    """

    batch = len(H['x'])
    H_o = np.zeros(batch, dtype=sim_specs['out'])

    for i, x in enumerate(H['x']):
        H_o['f'][i] = six_hump_camel_func(x)

        if 'grad' in H_o.dtype.names:
            H_o['grad'][i] = six_hump_camel_grad(x)

        if 'user' in sim_specs and 'pause_time' in sim_specs['user']:
            time.sleep(sim_specs['user']['pause_time'])

    return H_o, persis_info


def six_hump_camel_simple(x, persis_info, sim_specs, _):
    """
    Evaluates the six hump camel function for a single point ``x``.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    H_o = np.zeros(1, dtype=sim_specs['out'])

    H_o['f'] = six_hump_camel_func(x[0][0])

    if 'pause_time' in sim_specs['user']:
        time.sleep(sim_specs['user']['pause_time'])

    return H_o, persis_info


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2

    return term1 + term2 + term3


def six_hump_camel_grad(x):
    """
    Definition of the six-hump camel gradient
    """

    x1 = x[0]
    x2 = x[1]
    grad = np.zeros(2)

    grad[0] = 2.0*(x1**5 - 4.2*x1**3 + 4.0*x1 + 0.5*x2)
    grad[1] = x1 + 16*x2**3 - 8*x2

    return grad


if __name__ == "__main__":
    x = (float(sys.argv[1]), float(sys.argv[2]))
    result = six_hump_camel_func(x)
    print(result)
